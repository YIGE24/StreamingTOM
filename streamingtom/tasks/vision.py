import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from ..modules.streamingtom_context import StreamingTOMContext
from ..utils.profiler import get_profiler


class VisionTask:

    def __init__(self, ctr, oqm):
        self.ctr = ctr
        self.oqm = oqm

    def encode_vision_batch(self, frame_batch: Dict, vision_tower, mm_projector) -> Tuple[torch.Tensor, torch.Tensor]:
        profiler = get_profiler()
        frame_info = frame_batch.get('frame_info', {})

        frames = frame_batch['frames']
        grid_thw = frame_batch.get('grid_thw')
        device = vision_tower.device

        local_batch_idx = frame_info.get('local_batch_idx', 0)

        encoding_start_frame = frame_info.get('encoding_start_frame', None)
        if encoding_start_frame and encoding_start_frame > 0 and local_batch_idx == 0:
            assert batch_idx > 0, f"Incremental encoding with start_frame={encoding_start_frame} but batch_idx=0!"

        with torch.no_grad():
            if grid_thw is not None:
                batch_tensor = torch.cat(frames, dim=0).to(device)
                with profiler.timer('vision_tower'):
                    features_tuple, attention_scores = vision_tower(batch_tensor, grid_thw=grid_thw)
                processed_features = features_tuple[1] if isinstance(features_tuple, tuple) else features_tuple

                T = len(frames)
                _, H, W = grid_thw[0].tolist() if grid_thw.dim() > 1 else grid_thw.tolist()
                merge_size = getattr(vision_tower, 'spatial_merge_size', 2)
                merged_tokens = (H // merge_size) * (W // merge_size)
                processed_features = processed_features.view(T, merged_tokens, -1)
                if attention_scores is not None and attention_scores.numel() > 0:
                    pooled_scores = []
                    for t in range(T):
                        scores = attention_scores[t * H * W:(t + 1) * H * W].view(H, W)
                        pooled = F.avg_pool2d(scores.unsqueeze(0).unsqueeze(0).float(), kernel_size=merge_size, stride=merge_size).squeeze()
                        pooled_scores.append(pooled.flatten())
                    attention_scores = torch.cat(pooled_scores).view(T, merged_tokens)
            else:
                batch_tensor = torch.stack(frames, dim=0).to(device)
                with profiler.timer('vision_tower'):
                    raw_features, attention_scores, _ = vision_tower(batch_tensor)
                with profiler.timer('mm_projector'):
                    processed_features = mm_projector(raw_features)
                processed_features = self._apply_2d_pool(processed_features, stride=2)
                attention_scores = self._apply_2d_pool(attention_scores.unsqueeze(-1), stride=2).squeeze(-1)

        return processed_features, attention_scores

    def process_vision_batch(self, video_id: str, processed_features: torch.Tensor, current_state: Dict[str, Any],
                             attention_scores, batch_idx: int, model_forward_fn, frame_info: Dict[str, Any]) -> Dict[str, Any]:
        profiler = get_profiler()

        batch_type = frame_info.get('batch_type', {})
        is_last = batch_type.get('is_last', False)

        with profiler.timer('compress_features'):
            compressed_features, compression_info = self.ctr.compress_features(
                processed_features,
                current_state,
                attention_scores
            )

        streamingtom_ctx = StreamingTOMContext.get_instance()
        streamingtom_ctx.set_encode_mode(video_id, batch_idx)
        streamingtom_ctx.set_oqm(self.oqm)
        streamingtom_ctx.should_store_keys = True

        model = model_forward_fn.__self__ if hasattr(model_forward_fn, '__self__') else None
        assert model is not None, f"Cannot extract model from model_forward_fn"
        streamingtom_ctx.inject_to_model(model)

        if batch_idx > 0:
            accumulated_kv = self._get_accumulated_kv_cache(video_id, model)
            assert accumulated_kv is not None, f"Batch {batch_idx} should have accumulated KV cache"
        else:
            accumulated_kv = None

        with profiler.timer('prefill_kv'):
            prefill_result = self._prefill_vision_tokens(compressed_features, model_forward_fn, accumulated_kv)

        if prefill_result.get('past_key_values'):
            with profiler.timer('store_kv'):
                self._store_new_kv_cache(
                    video_id,
                    prefill_result['past_key_values'],
                    prefill_result.get('accumulated_length', 0)
                )

        del accumulated_kv, prefill_result
        streamingtom_ctx.should_store_keys = False

        if is_last:
            streamingtom_ctx.clear_mode()

        return {**current_state, 'ctr_state': compression_info['ctr_state']}

    def _get_accumulated_kv_cache(self, video_id: str, model):
        if not model:
            return None

        max_layers = StreamingTOMContext.get_model_num_layers(model)
        batch_idx = getattr(StreamingTOMContext.get_instance(), 'batch_idx', 0)

        if batch_idx > 0:
            stored_count = self.oqm.stored_tokens_count.get(video_id, {}).get(0, 0)
            assert stored_count > 0, f"Batch {batch_idx}: No stored tokens found"

        cache = []
        for layer_idx in range(max_layers):
            kv_pair = self.oqm.get_windowed_kv(video_id, layer_idx)
            assert kv_pair is not None, f"Layer {layer_idx} has no stored KV cache"

            if kv_pair is not None:
                k, v = kv_pair
                assert k is not None and v is not None, f"Layer {layer_idx}: KV is None"
                assert k.shape[2] > 0, f"Layer {layer_idx}: empty K cache"
                assert v.shape[2] > 0, f"Layer {layer_idx}: empty V cache"

            cache.append(kv_pair)

        assert len(cache) == max_layers, f"Expected {max_layers} layers, got {len(cache)}"

        if cache[0] is not None and cache[0][0] is not None:
            expected_tokens = cache[0][0].shape[2]
            assert expected_tokens > 0, "Cache has 0 tokens"
            for layer_idx in range(1, max_layers):
                actual_tokens = cache[layer_idx][0].shape[2]
                assert actual_tokens == expected_tokens, f"Layer {layer_idx}: token count mismatch"

        return cache

    def _store_new_kv_cache(self, video_id: str, past_key_values, accumulated_length: int):
        if not past_key_values or not hasattr(past_key_values, 'key_cache'):
            return

        num_layers = len(past_key_values.key_cache) if hasattr(past_key_values, 'key_cache') else 0
        assert num_layers > 0, "No key_cache layers"
        if num_layers == 0:
            return

        assert num_layers == 28, f"Expected 28 layers, got {num_layers}"
        assert len(past_key_values.value_cache) == 28, f"Value cache layers mismatch"

        stored_layers = 0
        for layer_idx, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            assert k is not None, f"Layer {layer_idx}: key is None"
            assert v is not None, f"Layer {layer_idx}: value is None"
            assert isinstance(k, torch.Tensor), f"Layer {layer_idx}: key not Tensor"
            assert isinstance(v, torch.Tensor), f"Layer {layer_idx}: value not Tensor"

            if accumulated_length > 0:
                assert k.shape[2] > accumulated_length, f"Layer {layer_idx}: insufficient tokens (got {k.shape[2]}, accumulated {accumulated_length})"

                original_length = k.shape[2]
                k, v = k[:, :, accumulated_length:, :], v[:, :, accumulated_length:, :]

                new_length = original_length - accumulated_length
                assert k.shape[2] == new_length, f"Layer {layer_idx}: slicing error (expected {new_length}, got {k.shape[2]})"

            assert k.shape[2] > 0, f"Layer {layer_idx}: no tokens to store"

            self.oqm.store_kv_cache(video_id, layer_idx, k, v)
            stored_layers += 1

        assert stored_layers == 28, f"Stored {stored_layers} layers, expected 28"

    def _prefill_vision_tokens(self, vision_context: torch.Tensor, model_forward_fn, accumulated_kv_cache=None) -> Dict[str, Any]:
        if vision_context is None or vision_context.numel() == 0:
            return {'past_key_values': accumulated_kv_cache}

        with torch.no_grad():
            streamingtom_ctx = StreamingTOMContext.get_instance()
            batch_idx = getattr(streamingtom_ctx, 'batch_idx', 0)

            if vision_context.dim() == 2:
                vision_context = vision_context.unsqueeze(0)
            batch_size, seq_len, _ = vision_context.shape
            device = vision_context.device
            model = model_forward_fn.__self__

            video_has_cache = self.oqm.get_batch_info(streamingtom_ctx.video_id)['total_tokens'] > 0
            should_init_prompt = (batch_idx == 0 and not video_has_cache)

            if should_init_prompt:
                streamingtom_ctx.should_store_keys = False
                accumulated_kv_cache = self._init_system_prompt(model, device)
                streamingtom_ctx.should_store_keys = True
            elif accumulated_kv_cache:
                accumulated_kv_cache = self._convert_to_dynamic_cache(accumulated_kv_cache)

            kv_cache_len = self._get_cache_length(accumulated_kv_cache)
            if kv_cache_len is None:
                kv_cache_len = 0
            attention_mask = torch.ones(batch_size, kv_cache_len + seq_len, dtype=torch.long, device=device)
            position_ids = torch.arange(kv_cache_len, kv_cache_len + seq_len, device=device).unsqueeze(0)
            if batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)

            actual_forward = (model.language_model.forward if hasattr(model, 'language_model')
                            else model.model.forward if hasattr(model, 'model')
                            else model_forward_fn)
            outputs = actual_forward(
                inputs_embeds=vision_context,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=accumulated_kv_cache,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False
            )

            if not outputs or not outputs.past_key_values:
                return {'past_key_values': None}

            assert hasattr(outputs.past_key_values, 'key_cache'), "No key_cache after forward"
            assert len(outputs.past_key_values.key_cache) == 28, f"Expected 28 layers, got {len(outputs.past_key_values.key_cache)}"

            for i, (k, v) in enumerate(zip(outputs.past_key_values.key_cache, outputs.past_key_values.value_cache)):
                assert k is not None, f"Layer {i} key is None"
                assert v is not None, f"Layer {i} value is None"
                assert k.shape[2] > 0, f"Layer {i} has 0 tokens"

            actual_accumulated_length = kv_cache_len

            actual_kv_len = outputs.past_key_values.key_cache[0].shape[2] if outputs.past_key_values.key_cache else 0
            expected_kv_len = actual_accumulated_length + vision_context.shape[1]
            assert actual_kv_len == expected_kv_len, f"KV length mismatch: {actual_kv_len} != {expected_kv_len}"

            return {
                'past_key_values': outputs.past_key_values,
                'accumulated_length': actual_accumulated_length
            }

    def _get_cache_length(self, cache):
        if not cache:
            return None

        if isinstance(cache, list):
            if not cache or cache[0] is None or cache[0][0] is None:
                return None
            return cache[0][0].shape[2]

        if hasattr(cache, 'get_seq_length'):
            return cache.get_seq_length()

        if hasattr(cache, 'key_cache') and cache.key_cache:
            first_layer = cache.key_cache[0]
            if first_layer is not None:
                return first_layer.shape[2]
            raise ValueError("Key cache first layer missing")

        raise ValueError(f"Unknown cache type {type(cache)}")

    def _convert_to_dynamic_cache(self, cache):
        if not cache:
            return None
        assert isinstance(cache, list), f"Cache must be a list, got {type(cache)}"
        assert len(cache) == 28, f"Cache must have 28 layers, got {len(cache)}"

        assert cache[0] is not None and cache[0][0] is not None, "Layer 0 cache is None"
        assert cache[0][0].shape[2] > 0, f"Layer 0 has 0 tokens"

        from transformers.cache_utils import DynamicCache
        cache_obj = DynamicCache()

        total_tokens_before = cache[0][0].shape[2]

        for layer_idx, (k, v) in enumerate(cache):
            assert k is not None, f"Layer {layer_idx} key is None"
            assert v is not None, f"Layer {layer_idx} value is None"
            assert k.shape[2] == total_tokens_before, f"Layer {layer_idx} token count mismatch"
            cache_obj.update(k, v, layer_idx)

        assert len(cache_obj.key_cache) == 28, f"Expected 28 layers, got {len(cache_obj.key_cache)}"
        assert cache_obj.key_cache[0] is not None, "DynamicCache key_cache[0] is None"
        assert cache_obj.key_cache[0].shape[2] == total_tokens_before, "Token count mismatch after conversion"

        return cache_obj

    def _init_system_prompt(self, model, device):
        init_ids = torch.tensor([[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198]], device=device)
        init_embeds = model.get_input_embeddings()(init_ids)
        forward_fn = (model.language_model.forward if hasattr(model, 'language_model')
                     else model.model.forward if hasattr(model, 'model') else model.forward)
        outputs = forward_fn(
            inputs_embeds=init_embeds,
            attention_mask=torch.ones_like(init_ids),
            position_ids=torch.arange(init_ids.shape[1], device=device).unsqueeze(0),
            use_cache=True,
            return_dict=True
        )

        assert outputs.past_key_values is not None, "No past_key_values in outputs"
        assert hasattr(outputs.past_key_values, 'key_cache'), "No key_cache in outputs"
        assert hasattr(outputs.past_key_values, 'value_cache'), "No value_cache in outputs"
        assert len(outputs.past_key_values.key_cache) == 28, f"Expected 28 layers, got {len(outputs.past_key_values.key_cache)}"

        streamingtom_ctx = StreamingTOMContext.get_instance()
        video_id = streamingtom_ctx.video_id
        assert video_id is not None, "video_id is required for storing system prompt"

        for layer_idx, (k, v) in enumerate(zip(outputs.past_key_values.key_cache, outputs.past_key_values.value_cache)):
            assert k.shape[2] == 14, f"Layer {layer_idx}: System prompt should be 14 tokens, got {k.shape[2]}"
            self.oqm.store_system_prompt(video_id, layer_idx, k, v)

        return outputs.past_key_values

    def _apply_2d_pool(self, features: torch.Tensor, stride: int = 2, mode: str = "bilinear") -> torch.Tensor:
        num_frames, num_tokens, hidden_dim = features.shape
        height = width = int(math.sqrt(num_tokens))
        assert height * width == num_tokens, "Token count must form a square grid"
        features = features.view(num_frames, height, width, hidden_dim).permute(0, 3, 1, 2).contiguous()

        if mode == "bilinear":
            features = F.interpolate(features, size=(math.ceil(height / stride), math.ceil(width / stride)), mode='bilinear')
        elif mode == "average":
            features = F.avg_pool2d(features, kernel_size=stride, stride=stride)
        elif mode == "max":
            features = F.max_pool2d(features, kernel_size=stride, stride=stride)
        else:
            raise ValueError(f"Unsupported pooling mode {mode} choose from 'bilinear', 'average', 'max'")
        _, _, new_h, new_w = features.shape
        return features.permute(0, 2, 3, 1).contiguous().view(num_frames, new_h * new_w, hidden_dim)

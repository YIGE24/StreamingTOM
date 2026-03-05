import os

import torch
from transformers.cache_utils import DynamicCache

from ..modules.streamingtom_context import StreamingTOMContext
from ..utils.profiler import get_profiler

PRE_QUESTION = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
POST_QUESTION = "<|im_end|>\n<|im_start|>assistant\n"

USE_FULL_PROMPT = os.getenv('STREAMINGTOM_USE_FULL_PROMPT') == '1'

IMAGE_TOKEN_INDEX = -200
MAX_TOKEN_ID = 200000


class QueryTask:

    def __init__(self, oqm, config):
        self.oqm = oqm
        self.config = config

    def prepare_retrieval_context(self, video_id: str, input_ids, model_forward_fn, tokenizer):
        batch_info = self.oqm.get_batch_info(video_id)
        total_vision_tokens = batch_info.get('total_tokens', 0)
        if total_vision_tokens == 0:
            return None

        model = model_forward_fn.__self__
        streamingtom_ctx = StreamingTOMContext.get_instance()

        assert streamingtom_ctx.mode != 'encode', f"Still in encode mode, batch_idx={streamingtom_ctx.batch_idx}"
        if streamingtom_ctx.mode is None and streamingtom_ctx.batch_idx != 0:
            assert False, f"batch_idx={streamingtom_ctx.batch_idx} but mode is None"

        streamingtom_ctx.retrieved_layers.clear()
        streamingtom_ctx.selected_vision_group_indices_per_layer.clear()
        streamingtom_ctx.mode = 'retrieve'
        streamingtom_ctx.video_id = video_id
        streamingtom_ctx.set_oqm(self.oqm)
        streamingtom_ctx.inject_to_model(model)

        clean_ids = self._clean_input_ids_for_retrieval(input_ids, tokenizer)

        streamingtom_ctx.retrieval_info = {
            'video_id': video_id,
            'budget': self.oqm.retrieval_max_tokens,
            'total_vision_tokens': total_vision_tokens,
            'oqm': self.oqm,
            'num_layers': StreamingTOMContext.get_model_num_layers(model),
            'clean_input_ids': clean_ids
        }

        return DynamicCache()

    def retrieve_and_generate(self, input_ids, model_forward_fn, tokenizer, **kwargs):
        profiler = get_profiler()
        profiler.new_sample()

        measure_ttft = os.getenv('STREAMINGTOM_MEASURE_TTFT', '0') == '1'
        if measure_ttft and torch.cuda.is_available():
            ttft_start_event = torch.cuda.Event(enable_timing=True)
            ttft_end_event = torch.cuda.Event(enable_timing=True)
            ttft_start_event.record()

        measure_memory = os.getenv('STREAMINGTOM_MEASURE_MEMORY', '0') == '1'
        if measure_memory and torch.cuda.is_available():
            torch.cuda.synchronize()

        model = model_forward_fn.__self__
        device = next(model.parameters()).device
        input_ids, original_input_ids = self._prepare_input_ids(input_ids, device)
        sampling_params = {
            k: kwargs.get(k)
            for k in ['temperature', 'do_sample', 'top_p', 'top_k', 'max_new_tokens']
        }

        with torch.no_grad():
            retrieved_past_kv = self._perform_retrieval_forward(model, self._get_retrieval_input_ids(input_ids, device))
            with profiler.timer('generate_first_token'):
                first_token, current_past_kv = self._generate_first_token(
                    model, original_input_ids, retrieved_past_kv, tokenizer, device, sampling_params)

            if measure_ttft and torch.cuda.is_available():
                ttft_end_event.record()
                torch.cuda.synchronize()
                ttft_ms = ttft_start_event.elapsed_time(ttft_end_event)
                ttft = ttft_ms / 1000.0
                print(f"[STREAMINGTOM TTFT] Time to First Token: {ttft:.4f} seconds")
                profiler.record_metric('ttft_sec', ttft)

                decode_start_event = torch.cuda.Event(enable_timing=True)
                decode_end_event = torch.cuda.Event(enable_timing=True)
                decode_start_event.record()

            if first_token == tokenizer.eos_token_id:
                result = self._cleanup_and_return([first_token], device, original_input_ids, model)
            else:
                with profiler.timer('generate_tokens'):
                    all_tokens = self._continue_generation(
                        model, [first_token], current_past_kv, tokenizer.eos_token_id, sampling_params, device)

                if measure_ttft and torch.cuda.is_available():
                    decode_end_event.record()
                    torch.cuda.synchronize()
                    decode_time_ms = decode_start_event.elapsed_time(decode_end_event)
                    decode_time = decode_time_ms / 1000.0
                    num_generated = len(all_tokens) - 1
                    if decode_time > 0 and num_generated > 0:
                        throughput = num_generated / decode_time
                        print(f"[STREAMINGTOM DECODE] Throughput: {throughput:.1f} tokens/sec")
                        profiler.record_metric('throughput', throughput)

                result = self._cleanup_and_return(all_tokens, device, original_input_ids, model)

        if measure_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[STREAMINGTOM MEMORY] Peak: {peak_memory_gb:.2f} GB")
            profiler.record_metric('memory_gb', peak_memory_gb)

        return result

    def _prepare_input_ids(self, input_ids, device):
        if not hasattr(input_ids, 'to'):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        else:
            input_ids = input_ids.to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if (input_ids == IMAGE_TOKEN_INDEX).any():
            mask = input_ids != IMAGE_TOKEN_INDEX
            input_ids = input_ids[:, mask[0]] if mask[0].any() else input_ids

        return input_ids, input_ids.clone()

    def _get_retrieval_input_ids(self, input_ids, device):
        streamingtom_ctx = StreamingTOMContext.get_instance()

        if not (streamingtom_ctx.retrieval_info and 'clean_input_ids' in streamingtom_ctx.retrieval_info):
            return input_ids

        clean_ids = streamingtom_ctx.retrieval_info['clean_input_ids']
        assert len(clean_ids) > 0, "clean_input_ids is empty"

        if not hasattr(clean_ids, 'to'):
            clean_ids = torch.tensor(clean_ids, dtype=torch.long, device=device)
        else:
            clean_ids = clean_ids.to(device)

        assert (clean_ids != IMAGE_TOKEN_INDEX).all(), "clean_input_ids contains IMAGE_TOKEN"

        if clean_ids.dim() == 1:
            clean_ids = clean_ids.unsqueeze(0)

        return clean_ids

    def _perform_retrieval_forward(self, model, retrieval_input_ids):
        profiler = get_profiler()
        streamingtom_ctx = StreamingTOMContext.get_instance()
        assert streamingtom_ctx.video_id is not None, "video_id not set"
        assert streamingtom_ctx.mode == 'retrieve', f"Wrong mode: {streamingtom_ctx.mode}"
        assert hasattr(streamingtom_ctx, 'retrieval_info'), "No retrieval_info"

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            first_layer = model.model.layers[0]
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            first_layer = model.language_model.layers[0]
        else:
            first_layer = None

        assert first_layer is not None and hasattr(first_layer.self_attn, '_streamingtom_context'), "Model not injected"

        streamingtom_ctx.should_store_keys = False

        with profiler.timer('retrieval_forward'):
            if hasattr(model, 'language_model'):
                _ = model.language_model(
                    input_ids=retrieval_input_ids,
                    past_key_values=DynamicCache(),
                    use_cache=False,
                    return_dict=True
                )
            else:
                inputs_embeds = model.get_input_embeddings()(retrieval_input_ids)
                _ = model.model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=DynamicCache(),
                    use_cache=False,
                    return_dict=True
                )

        video_id = streamingtom_ctx.video_id
        retrieved_kv_cache = DynamicCache()

        with profiler.timer('reconstruct_kv'):
            assert self.oqm is not None, "OQM is None"
            assert video_id is not None, "video_id is None"

            num_layers = StreamingTOMContext.get_model_num_layers(model)
            assert num_layers == 28, f"Expected 28 layers, got {num_layers}"

            assert hasattr(streamingtom_ctx, 'selected_vision_group_indices_per_layer'), "No selected indices"
            assert len(streamingtom_ctx.selected_vision_group_indices_per_layer) > 0, "Empty selected indices"

            layers_with_kv = 0
            for layer_idx in range(num_layers):
                layer_kv = self._get_retrieved_kv_for_layer(video_id, layer_idx)
                assert layer_kv is not None, f"Layer {layer_idx}: No KV"

                k, v = layer_kv
                assert k is not None and v is not None, f"Layer {layer_idx}: None KV"
                assert k.shape == v.shape, f"Layer {layer_idx}: shape mismatch"
                assert k.shape[2] > 0, f"Layer {layer_idx}: empty KV"

                min_expected_length = self.oqm.init_token_count + self.oqm.group_size
                assert k.shape[2] > min_expected_length, f"Layer {layer_idx}: KV too short"

                retrieved_kv_cache.update(k, v, layer_idx)
                layers_with_kv += 1

            assert layers_with_kv == num_layers, f"Missing layers: {layers_with_kv}/{num_layers}"

        streamingtom_ctx.clear_mode()

        assert retrieved_kv_cache is not None, "No retrieved KV cache"
        if hasattr(retrieved_kv_cache, 'get_seq_length'):
            seq_len = retrieved_kv_cache.get_seq_length()
            assert seq_len > 0, f"Empty KV cache: {seq_len}"

        return retrieved_kv_cache

    def _get_retrieved_kv_for_layer(self, video_id: str, layer_idx: int):
        streamingtom_ctx = StreamingTOMContext.get_instance()

        if not hasattr(streamingtom_ctx, 'selected_vision_group_indices_per_layer'):
            return None
        if layer_idx not in streamingtom_ctx.selected_vision_group_indices_per_layer:
            return None

        selected_vision_group_indices = streamingtom_ctx.selected_vision_group_indices_per_layer[layer_idx]

        video_kv = self.oqm.get_selective_kv(
            video_id,
            layer_idx,
            selected_vision_group_indices
        )

        return video_kv

    def _generate_first_token(self, model, original_input_ids, retrieved_past_kv,
                              tokenizer, device, sampling_params):
        formatted_ids = self._format_input_ids_for_generation(original_input_ids, tokenizer, device)
        inputs_embeds = model.get_input_embeddings()(formatted_ids)

        past_kv_len = (retrieved_past_kv.get_seq_length() if hasattr(retrieved_past_kv, 'get_seq_length')
                      else retrieved_past_kv.key_cache[0].shape[2] if hasattr(retrieved_past_kv, 'key_cache') and retrieved_past_kv.key_cache[0] is not None
                      else 0)

        assert formatted_ids.shape[1] > 0, "Empty formatted input"
        assert past_kv_len > 0, "No past KV cache"

        if hasattr(retrieved_past_kv, 'key_cache') and retrieved_past_kv.key_cache:
            first_layer_k = retrieved_past_kv.key_cache[0]
            assert first_layer_k.shape[2] == past_kv_len, f"KV length mismatch: {first_layer_k.shape[2]} != {past_kv_len}"
            assert past_kv_len >= 14, f"KV cache too small: {past_kv_len}"

        batch_size, query_len = inputs_embeds.shape[:2]
        attention_mask = torch.ones((batch_size, past_kv_len + query_len), dtype=torch.long, device=inputs_embeds.device)
        position_ids = torch.arange(past_kv_len, past_kv_len + query_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        assert attention_mask.shape[1] == past_kv_len + query_len, f"Attention mask mismatch"
        assert position_ids.shape[1] == query_len, f"Position IDs mismatch"

        forward_fn = (model.language_model if hasattr(model, 'language_model')
                     else model.model if hasattr(model, 'model')
                     else model)

        outputs = forward_fn(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=retrieved_past_kv, use_cache=True, return_dict=True)

        logits = model.lm_head(outputs.last_hidden_state) if hasattr(model, 'lm_head') else outputs.logits
        next_token = self._sample_token(logits[0, -1, :], **sampling_params)

        assert next_token is not None, "No token generated"

        return next_token, outputs.past_key_values

    def _continue_generation(self, model, generated_tokens, current_past_kv,
                            eos_token_id, sampling_params, device):
        max_new_tokens = sampling_params.get('max_new_tokens', 128)
        for _ in range(1, max_new_tokens):
            outputs = model(input_ids=torch.tensor([[generated_tokens[-1]]], device=device),
                          past_key_values=current_past_kv, use_cache=True, return_dict=True)
            next_token = self._sample_token(outputs.logits[0, -1, :], **sampling_params)
            generated_tokens.append(next_token)
            if next_token == eos_token_id:
                break
            current_past_kv = outputs.past_key_values
        return generated_tokens

    def _cleanup_and_return(self, generated_tokens, device, original_input_ids=None, model=None):
        StreamingTOMContext.get_instance().clear_mode()

        if original_input_ids is not None and model is not None:
            model_class = model.__class__.__name__.lower()
            is_llava = 'llava' in model_class or 'onevision' in model_class
            if is_llava:
                return torch.tensor([generated_tokens], dtype=torch.long, device=device)
            else:
                full_sequence = original_input_ids[0].tolist() + generated_tokens
                return torch.tensor([full_sequence], dtype=torch.long, device=device)

        return torch.tensor([generated_tokens], dtype=torch.long, device=device)

    def _clean_input_ids_for_retrieval(self, input_ids, tokenizer):
        text = self._decode_input_ids(input_ids, tokenizer)
        question = self._extract_pure_question(text)
        return tokenizer(question, return_tensors="pt").input_ids

    def _format_input_ids_for_generation(self, input_ids, tokenizer, device):
        text = self._decode_input_ids(input_ids, tokenizer)
        question = self._extract_pure_question(text)

        if USE_FULL_PROMPT:
            formatted_prompt = f"{PRE_QUESTION}{question}{POST_QUESTION}"
        else:
            formatted_prompt = f"{question}{POST_QUESTION}"

        return tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)

    def _decode_input_ids(self, input_ids, tokenizer):
        if hasattr(input_ids, 'cpu'):
            input_ids = (input_ids[0] if input_ids.dim() == 2 else input_ids).cpu().tolist()
        elif hasattr(input_ids, '__len__') and len(input_ids) > 0:
            input_ids = input_ids[0] if isinstance(input_ids[0], list) else input_ids

        filtered_ids = [token_id for token_id in input_ids if 0 <= token_id < MAX_TOKEN_ID]
        return tokenizer.decode(filtered_ids, skip_special_tokens=False)

    def _extract_pure_question(self, text):
        assert PRE_QUESTION in text, "Missing PRE_QUESTION"
        assert POST_QUESTION in text, "Missing POST_QUESTION"

        start = text.find(PRE_QUESTION) + len(PRE_QUESTION)
        end = text.find(POST_QUESTION, start)
        assert end > start, "Invalid question format"

        question = text[start:end]
        assert question, "Empty question"
        return question.strip()

    def _sample_token(self, logits, **sampling_params):
        if not sampling_params.get('do_sample'):
            return torch.argmax(logits, dim=-1).item()

        if (temp := sampling_params.get('temperature')) and temp != 1.0:
            logits = logits / temp

        probs = torch.softmax(logits, dim=-1)

        if (top_k := sampling_params.get('top_k')) and top_k > 0:
            top_k = min(top_k, probs.size(-1))
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter(-1, top_k_indices, top_k_probs)

        if (top_p := sampling_params.get('top_p')) and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            probs[sorted_indices[sorted_indices_to_remove]] = 0

        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=1).item()

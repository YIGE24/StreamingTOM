import torch
from typing import Dict, Any, Tuple, Optional
from ..main import get_streamingtom_config


class OQM:

    def __init__(self):
        config = get_streamingtom_config().config

        self.retrieval_max_tokens = config['oqm_retrieval_max_tokens']
        self.enable_quantization = config['oqm_enable_quantization']
        self.group_size = config['oqm_group_size']
        self.init_token_count = config['oqm_init_token_count']
        self.sliding_window_size = config['oqm_sliding_window_size']
        self.quantization_bits = config['oqm_quantization_bits']
        self.quantization_levels = 2 ** self.quantization_bits - 1
        self.pack_size = 8 // self.quantization_bits
        assert self.group_size % self.pack_size == 0

        self.kv_cache_storage: Dict = {}
        self.quantized_storage: Dict = {}
        self.batch_metadata: Dict = {}
        self.group_keys: Dict = {}
        self.stored_tokens_count: Dict = {}
        self.original_dtype: Dict = {}

    def store_system_prompt(self, video_id: str, layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor):
        self._ensure_storage_initialized(video_id, layer_idx)
        assert key_cache.shape == value_cache.shape, f"Layer {layer_idx}: K/V shape mismatch"
        assert key_cache.shape[2] == self.init_token_count, f"Layer {layer_idx}: unexpected prompt length"

        self.original_dtype.setdefault(video_id, {})[layer_idx] = key_cache.dtype

        if self.enable_quantization:
            storage_dict = self.quantized_storage[video_id][layer_idx]
            storage_dict['init_tokens'] = (
                key_cache.detach(),
                value_cache.detach()
            )
        else:
            cache = self.kv_cache_storage[video_id]
            existing_entry = cache.get(layer_idx)
            assert existing_entry is None, f"Layer {layer_idx}: system prompt already stored"
            cache[layer_idx] = (key_cache.detach(), value_cache.detach())

        self.stored_tokens_count[video_id][layer_idx] = self.init_token_count

        if layer_idx == 0:
            self.batch_metadata[video_id].append({
                'batch_idx': 0,
                'token_count': self.init_token_count
            })

    def store_kv_cache(self, video_id: str, layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor):
        self._ensure_storage_initialized(video_id, layer_idx)
        assert key_cache.shape == value_cache.shape, f"Layer {layer_idx}: K/V shape mismatch"

        new_tokens = key_cache.shape[2]
        assert new_tokens > 0, f"Layer {layer_idx}: No tokens to store"
        existing_length = self.stored_tokens_count[video_id].get(layer_idx, 0)

        assert existing_length >= self.init_token_count, f"Layer {layer_idx}: system prompt missing"

        self.original_dtype.setdefault(video_id, {})[layer_idx] = key_cache.dtype

        if self.enable_quantization:
            storage_dict = self.quantized_storage[video_id][layer_idx]
            assert 'init_tokens' in storage_dict, f"Layer {layer_idx}: init tokens missing"
            self._quantize_new_tokens(storage_dict, key_cache, value_cache, 0)
        else:
            cache = self.kv_cache_storage[video_id]
            new_k, new_v = key_cache.detach(), value_cache.detach()
            if existing_length > 0:
                old_k, old_v = cache[layer_idx]
                cache[layer_idx] = (torch.cat([old_k, new_k], dim=2), torch.cat([old_v, new_v], dim=2))
            else:
                cache[layer_idx] = (new_k, new_v)

        self.stored_tokens_count[video_id][layer_idx] = existing_length + new_tokens

        if layer_idx == 0:
            self.batch_metadata[video_id].append({
                'batch_idx': len(self.batch_metadata[video_id]),
                'token_count': self.stored_tokens_count[video_id][layer_idx]
            })

    def store_token_keys_as_groups(self, video_id: str, layer_idx: int, token_level_keys: torch.Tensor):
        self.group_keys.setdefault(video_id, {})
        num_tokens = token_level_keys.shape[0]

        assert num_tokens % self.group_size == 0, f"Layer {layer_idx}: token count not aligned"
        assert token_level_keys.dim() == 2, f"Expected 2D tensor, got {token_level_keys.dim()}D"
        assert not torch.isnan(token_level_keys).any(), f"Layer {layer_idx}: NaN in keys"

        num_groups = num_tokens // self.group_size
        group_keys_new = token_level_keys.reshape(num_groups, self.group_size, -1).mean(dim=1).detach()

        if layer_idx in self.group_keys[video_id]:
            existing_groups = self.group_keys[video_id][layer_idx]
            assert existing_groups.shape[1] == group_keys_new.shape[1], f"Layer {layer_idx}: key dimension mismatch"
            self.group_keys[video_id][layer_idx] = torch.cat([existing_groups, group_keys_new], dim=0)
        else:
            self.group_keys[video_id][layer_idx] = group_keys_new

    def get_windowed_kv(self, video_id: str, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        storage = self.quantized_storage if self.enable_quantization else self.kv_cache_storage
        if video_id not in storage or layer_idx not in storage[video_id]:
            return None

        if self.enable_quantization:
            target_dtype = self.original_dtype.get(video_id, {}).get(layer_idx)
            return self._reconstruct_windowed_kv(storage[video_id][layer_idx], target_dtype)
        return storage[video_id][layer_idx]

    def get_group_keys(self, video_id: str, layer_idx: int) -> Optional[torch.Tensor]:
        if video_id not in self.group_keys: return None
        if layer_idx not in self.group_keys[video_id]: return None

        group_keys = self.group_keys[video_id][layer_idx]

        if video_id in self.stored_tokens_count and layer_idx in self.stored_tokens_count[video_id]:
            total_stored = self.stored_tokens_count[video_id][layer_idx]
            vision_tokens = total_stored - self.init_token_count
            expected_groups = vision_tokens // self.group_size
            actual_groups = group_keys.shape[0]
            assert actual_groups == expected_groups, f"Layer {layer_idx}: group count mismatch"

        return group_keys

    def get_batch_info(self, video_id: str) -> Dict[str, Any]:
        if video_id not in self.batch_metadata or not self.batch_metadata[video_id]:
            return {'batch_idx': 0, 'total_tokens': 0}
        last_batch = self.batch_metadata[video_id][-1]
        return {
            'batch_idx': len(self.batch_metadata[video_id]),
            'total_tokens': last_batch['token_count']
        }

    def get_selective_kv(self, video_id: str, layer_idx: int, selected_vision_group_indices: torch.Tensor):
        storage = self.quantized_storage if self.enable_quantization else self.kv_cache_storage
        assert video_id in storage, f"Video {video_id} not found"
        assert layer_idx in storage[video_id], f"Layer {layer_idx} not found"
        assert selected_vision_group_indices is not None and len(selected_vision_group_indices) > 0

        device = selected_vision_group_indices.device
        init_count = self.init_token_count

        total_stored = self.stored_tokens_count.get(video_id, {}).get(layer_idx, 0)

        total_vision_tokens = total_stored - init_count
        total_vision_groups = total_vision_tokens // self.group_size
        valid_range = ((selected_vision_group_indices >= 0).all() and (selected_vision_group_indices < total_vision_groups).all())
        assert valid_range, f"Layer {layer_idx}: group index out of range"

        init_indices = torch.arange(init_count, device=device, dtype=torch.long)
        group_starts = init_count + selected_vision_group_indices * self.group_size
        group_indices = group_starts.unsqueeze(1) + torch.arange(self.group_size, device=device)
        group_indices = group_indices.flatten()
        token_indices = torch.cat([init_indices, group_indices])

        unique_ok = len(torch.unique(selected_vision_group_indices)) == len(selected_vision_group_indices)
        assert unique_ok, f"Layer {layer_idx}: duplicate group index"
        sorted_ok = torch.all(selected_vision_group_indices[1:] > selected_vision_group_indices[:-1])
        assert sorted_ok, f"Layer {layer_idx}: group indices not sorted"
        assert token_indices.max() < total_stored, f"Layer {layer_idx}: token index out of bounds"

        if self.enable_quantization:
            storage_dict = self.quantized_storage[video_id][layer_idx]
            assert 'init_tokens' in storage_dict, f"Layer {layer_idx}: Missing init_tokens"

            target_dtype = self.original_dtype.get(video_id, {}).get(layer_idx, torch.float16)
            result = self._reconstruct_selective_kv(storage_dict, token_indices, target_dtype)
        else:
            kv = self.kv_cache_storage[video_id][layer_idx]
            assert kv is not None, f"Layer {layer_idx}: KV cache is None!"
            assert kv[0] is not None and kv[1] is not None, f"Layer {layer_idx}: K or V is None!"
            result = (kv[0][:, :, token_indices, :], kv[1][:, :, token_indices, :])

        return result

    def clear_cache(self, video_id: Optional[str] = None):
        storages = [self.kv_cache_storage, self.quantized_storage, self.batch_metadata,
                   self.group_keys, self.stored_tokens_count, self.original_dtype]

        if video_id is None:
            for storage in storages:
                storage.clear()
        else:
            for storage in storages:
                storage.pop(video_id, None)

        from ..modules.streamingtom_context import StreamingTOMContext
        streamingtom_ctx = StreamingTOMContext.get_instance()
        if streamingtom_ctx.video_id == video_id or video_id is None:
            streamingtom_ctx.clear_mode()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_storage_initialized(self, video_id: str, layer_idx: int):
        self.batch_metadata.setdefault(video_id, [])
        self.stored_tokens_count.setdefault(video_id, {})

        if self.enable_quantization:
            self.quantized_storage.setdefault(video_id, {}).setdefault(layer_idx, {
                'init_tokens': None,
                'keys_packed': [], 'keys_scales': [], 'keys_mins': [], 'keys_S': [],
                'values_packed': [], 'values_scales': [], 'values_mins': [], 'values_S': []
            })
        else:
            self.kv_cache_storage.setdefault(video_id, {}).setdefault(layer_idx, None)

    def _quantize_new_tokens(self, storage_dict: Dict, key_cache: torch.Tensor, value_cache: torch.Tensor, start: int):
        new_keys = key_cache[:, :, start:, :].contiguous()
        new_values = value_cache[:, :, start:, :].contiguous()
        new_len = new_keys.shape[2]
        assert new_len % self.group_size == 0

        k_packed, k_scales, k_mins, k_S = self._quantize_tensor(new_keys)
        v_packed, v_scales, v_mins, v_S = self._quantize_tensor(new_values)

        storage_dict['keys_packed'].append(k_packed)
        storage_dict['keys_scales'].append(k_scales)
        storage_dict['keys_mins'].append(k_mins)
        storage_dict['keys_S'].append(k_S)

        storage_dict['values_packed'].append(v_packed)
        storage_dict['values_scales'].append(v_scales)
        storage_dict['values_mins'].append(v_mins)
        storage_dict['values_S'].append(v_S)

    def _pack_Nbit(self, tensor: torch.Tensor) -> torch.Tensor:
        *batch_dims, N = tensor.shape
        assert N % self.pack_size == 0
        tensor = tensor.reshape(*batch_dims, N // self.pack_size, self.pack_size)
        shift_amounts = torch.arange(self.pack_size, device=tensor.device) * self.quantization_bits
        packed = (tensor << shift_amounts).sum(dim=-1).to(torch.uint8)
        return packed

    def _unpack_Nbit(self, packed: torch.Tensor) -> torch.Tensor:
        *batch_dims, N = packed.shape
        mask = (1 << self.quantization_bits) - 1
        shift_amounts = torch.arange(self.pack_size, device=packed.device) * self.quantization_bits
        packed_expanded = packed.unsqueeze(-1).expand(*batch_dims, N, self.pack_size)
        unpacked = (packed_expanded >> shift_amounts) & mask
        return unpacked.reshape(*batch_dims, N * self.pack_size)

    def _quantize_tensor(self, tensor: torch.Tensor) -> Tuple:
        B, H, S, D = tensor.shape
        num_groups = S // self.group_size

        tensor_flat = tensor.permute(0, 1, 3, 2).contiguous()
        tensor_reshaped = tensor_flat.reshape(B * H * D, num_groups, self.group_size)

        mins = tensor_reshaped.min(dim=-1)[0]
        maxs = tensor_reshaped.max(dim=-1)[0]
        scales = (maxs - mins) / self.quantization_levels
        scales = torch.clamp(scales, min=1e-8)

        tensor_normalized = (tensor_reshaped - mins.unsqueeze(-1)) / scales.unsqueeze(-1)
        tensor_quantized = tensor_normalized.clamp(0, self.quantization_levels).round_().to(torch.uint8)
        tensor_quantized = tensor_quantized.reshape(B, H, D, S)

        tensor_packed = self._pack_Nbit(tensor_quantized)
        scales = scales.reshape(B, H, D, num_groups).permute(0, 1, 3, 2)
        mins = mins.reshape(B, H, D, num_groups).permute(0, 1, 3, 2)
        tensor_packed = tensor_packed.permute(0, 1, 3, 2).contiguous()

        return tensor_packed, scales, mins, S

    def _dequantize_tensor(self, quantized_data: Tuple) -> torch.Tensor:
        tensor_packed, scales, mins, original_S = quantized_data
        B, H, _, D = tensor_packed.shape
        num_groups = original_S // self.group_size
        target_dtype = scales.dtype

        tensor_packed = tensor_packed.permute(0, 1, 3, 2).contiguous()
        scales = scales.permute(0, 1, 3, 2)
        mins = mins.permute(0, 1, 3, 2)

        tensor_unpacked = self._unpack_Nbit(tensor_packed)
        tensor_flat = tensor_unpacked.reshape(B * H * D, num_groups, self.group_size).to(target_dtype)
        scales_flat = scales.reshape(B * H * D, num_groups)
        mins_flat = mins.reshape(B * H * D, num_groups)
        tensor_dequant = tensor_flat * scales_flat.unsqueeze(-1) + mins_flat.unsqueeze(-1)

        tensor_dequant = tensor_dequant.reshape(B, H, D, original_S)
        tensor_dequant = tensor_dequant.permute(0, 1, 3, 2).contiguous()
        return tensor_dequant

    def _reconstruct_selective_kv(self, data: Dict, indices: torch.Tensor, target_dtype: torch.dtype) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        device = indices.device
        num_indices = len(indices)

        if data.get('keys_packed') and len(data['keys_packed']) > 0:
            B, H, _, D = data['keys_packed'][0].shape
        elif data['init_tokens']:
            B, H, _, D = data['init_tokens'][0].shape
        else:
            return None

        k_result = torch.zeros(B, H, num_indices, D, dtype=target_dtype, device=device)
        v_result = torch.zeros(B, H, num_indices, D, dtype=target_dtype, device=device)

        init_count = 0
        if data['init_tokens']:
            init_count = data['init_tokens'][0].shape[2]
            k_init, v_init = data['init_tokens']
            k_init = k_init.to(device)
            v_init = v_init.to(device)

            init_mask = indices < init_count
            if init_mask.any():
                init_indices_local = indices[init_mask]
                init_positions = torch.where(init_mask)[0]
                k_result[:, :, init_positions, :] = k_init[:, :, init_indices_local, :]
                v_result[:, :, init_positions, :] = v_init[:, :, init_indices_local, :]

        if data.get('keys_packed') and len(data['keys_packed']) > 0:
            num_batches = len(data['keys_S'])

            batch_sizes = torch.tensor(data['keys_S'], device=device, dtype=torch.long)
            batch_starts = torch.zeros(num_batches + 1, device=device, dtype=torch.long)
            batch_starts[0] = init_count
            batch_starts[1:] = init_count + batch_sizes.cumsum(0)

            quant_mask = indices >= init_count
            if quant_mask.any():
                quant_indices = indices[quant_mask]
                quant_positions = torch.where(quant_mask)[0]

                keys_packed_merged = torch.cat([data['keys_packed'][i].to(device) for i in range(num_batches)], dim=2)
                keys_scales_merged = torch.cat([data['keys_scales'][i].to(device) for i in range(num_batches)], dim=2)
                keys_mins_merged = torch.cat([data['keys_mins'][i].to(device) for i in range(num_batches)], dim=2)
                values_packed_merged = torch.cat([data['values_packed'][i].to(device) for i in range(num_batches)], dim=2)
                values_scales_merged = torch.cat([data['values_scales'][i].to(device) for i in range(num_batches)], dim=2)
                values_mins_merged = torch.cat([data['values_mins'][i].to(device) for i in range(num_batches)], dim=2)

                groups_per_batch = batch_sizes // self.group_size
                batch_group_offsets = torch.zeros(num_batches + 1, device=device, dtype=torch.long)
                batch_group_offsets[1:] = groups_per_batch.cumsum(0)

                batch_ids = torch.searchsorted(batch_starts[1:], quant_indices, right=False)

                local_indices = quant_indices - batch_starts[batch_ids]
                local_group_ids = local_indices // self.group_size
                within_group_ids = local_indices % self.group_size
                global_group_ids = batch_group_offsets[batch_ids] + local_group_ids

                unique_groups, inverse_indices = torch.unique(global_group_ids, return_inverse=True)

                if len(unique_groups) > 0:
                    packed_group_size = self.group_size // self.pack_size
                    num_unique_groups = len(unique_groups)

                    group_starts = unique_groups * packed_group_size
                    all_packed_indices = (group_starts.unsqueeze(1) +
                                        torch.arange(packed_group_size, device=device)).flatten()

                    k_packed_all = keys_packed_merged[:, :, all_packed_indices, :].reshape(
                        B, H, num_unique_groups, packed_group_size, D
                    )
                    v_packed_all = values_packed_merged[:, :, all_packed_indices, :].reshape(
                        B, H, num_unique_groups, packed_group_size, D
                    )

                    k_scales_all = keys_scales_merged[:, :, unique_groups, :]
                    k_mins_all = keys_mins_merged[:, :, unique_groups, :]
                    v_scales_all = values_scales_merged[:, :, unique_groups, :]
                    v_mins_all = values_mins_merged[:, :, unique_groups, :]

                    k_packed_reshaped = k_packed_all.permute(0, 1, 2, 4, 3).reshape(
                        B * H * num_unique_groups * D, packed_group_size
                    )
                    v_packed_reshaped = v_packed_all.permute(0, 1, 2, 4, 3).reshape(
                        B * H * num_unique_groups * D, packed_group_size
                    )

                    k_unpacked_flat = self._unpack_Nbit(k_packed_reshaped.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                    v_unpacked_flat = self._unpack_Nbit(v_packed_reshaped.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

                    k_unpacked = k_unpacked_flat.reshape(
                        B, H, num_unique_groups, D, self.group_size
                    ).permute(0, 1, 2, 4, 3)
                    v_unpacked = v_unpacked_flat.reshape(
                        B, H, num_unique_groups, D, self.group_size
                    ).permute(0, 1, 2, 4, 3)

                    k_dequant = k_unpacked.to(k_scales_all.dtype) * k_scales_all.unsqueeze(3) + k_mins_all.unsqueeze(3)
                    v_dequant = v_unpacked.to(v_scales_all.dtype) * v_scales_all.unsqueeze(3) + v_mins_all.unsqueeze(3)

                    k_result[:, :, quant_positions, :] = k_dequant[:, :, inverse_indices, within_group_ids, :]
                    v_result[:, :, quant_positions, :] = v_dequant[:, :, inverse_indices, within_group_ids, :]

        return k_result, v_result

    def _reconstruct_windowed_kv(self, data: Dict, target_dtype: torch.dtype) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        k_parts, v_parts = [], []

        if data.get('init_tokens'):
            init_k, init_v = data['init_tokens']
            k_parts.append(init_k)
            v_parts.append(init_v)

        if data.get('keys_packed') and data['keys_packed']:
            batch_sizes = data['keys_S']
            num_batches = len(batch_sizes)
            total_S = sum(batch_sizes)

            if total_S <= self.sliding_window_size:
                if num_batches > 1:
                    k_packed = torch.cat(data['keys_packed'], dim=2)
                    k_scales = torch.cat(data['keys_scales'], dim=2)
                    k_mins = torch.cat(data['keys_mins'], dim=2)
                    v_packed = torch.cat(data['values_packed'], dim=2)
                    v_scales = torch.cat(data['values_scales'], dim=2)
                    v_mins = torch.cat(data['values_mins'], dim=2)
                else:
                    k_packed = data['keys_packed'][0]
                    k_scales = data['keys_scales'][0]
                    k_mins = data['keys_mins'][0]
                    v_packed = data['values_packed'][0]
                    v_scales = data['values_scales'][0]
                    v_mins = data['values_mins'][0]

                k_parts.append(self._dequantize_tensor((k_packed, k_scales, k_mins, total_S)))
                v_parts.append(self._dequantize_tensor((v_packed, v_scales, v_mins, total_S)))
            else:
                tokens_needed = self.sliding_window_size
                cumsum_reverse = 0
                full_batches_from_end = []
                partial_batch_idx = -1
                partial_groups_needed = 0

                for idx in range(num_batches - 1, -1, -1):
                    batch_S = batch_sizes[idx]
                    if cumsum_reverse + batch_S <= tokens_needed:
                        full_batches_from_end.append(idx)
                        cumsum_reverse += batch_S
                    else:
                        tokens_from_this = tokens_needed - cumsum_reverse
                        partial_groups_needed = tokens_from_this // self.group_size
                        if partial_groups_needed > 0:
                            partial_batch_idx = idx
                        break

                if partial_batch_idx >= 0 and partial_groups_needed > 0:
                    batch_S = batch_sizes[partial_batch_idx]
                    total_groups = batch_S // self.group_size
                    start_group = total_groups - partial_groups_needed

                    start_packed_idx = (start_group * self.group_size) // self.pack_size
                    k_sliced = data['keys_packed'][partial_batch_idx][:, :, start_packed_idx:, :]
                    k_scales_sliced = data['keys_scales'][partial_batch_idx][:, :, start_group:, :]
                    k_mins_sliced = data['keys_mins'][partial_batch_idx][:, :, start_group:, :]

                    v_sliced = data['values_packed'][partial_batch_idx][:, :, start_packed_idx:, :]
                    v_scales_sliced = data['values_scales'][partial_batch_idx][:, :, start_group:, :]
                    v_mins_sliced = data['values_mins'][partial_batch_idx][:, :, start_group:, :]

                    sliced_S = partial_groups_needed * self.group_size

                    k_parts.append(self._dequantize_tensor((k_sliced, k_scales_sliced, k_mins_sliced, sliced_S)))
                    v_parts.append(self._dequantize_tensor((v_sliced, v_scales_sliced, v_mins_sliced, sliced_S)))

                if full_batches_from_end:
                    full_batches_from_end.reverse()

                    if len(full_batches_from_end) > 1:
                        k_full_packed = torch.cat([data['keys_packed'][i] for i in full_batches_from_end], dim=2)
                        k_full_scales = torch.cat([data['keys_scales'][i] for i in full_batches_from_end], dim=2)
                        k_full_mins = torch.cat([data['keys_mins'][i] for i in full_batches_from_end], dim=2)

                        v_full_packed = torch.cat([data['values_packed'][i] for i in full_batches_from_end], dim=2)
                        v_full_scales = torch.cat([data['values_scales'][i] for i in full_batches_from_end], dim=2)
                        v_full_mins = torch.cat([data['values_mins'][i] for i in full_batches_from_end], dim=2)

                        full_S = sum(batch_sizes[i] for i in full_batches_from_end)
                    else:
                        idx = full_batches_from_end[0]
                        k_full_packed = data['keys_packed'][idx]
                        k_full_scales = data['keys_scales'][idx]
                        k_full_mins = data['keys_mins'][idx]

                        v_full_packed = data['values_packed'][idx]
                        v_full_scales = data['values_scales'][idx]
                        v_full_mins = data['values_mins'][idx]

                        full_S = batch_sizes[idx]

                    k_parts.append(self._dequantize_tensor((k_full_packed, k_full_scales, k_full_mins, full_S)))
                    v_parts.append(self._dequantize_tensor((v_full_packed, v_full_scales, v_full_mins, full_S)))

        if not k_parts:
            return None

        k_final = torch.cat(k_parts, dim=2) if len(k_parts) > 1 else k_parts[0]
        v_final = torch.cat(v_parts, dim=2) if len(v_parts) > 1 else v_parts[0]

        total_vision_tokens = sum(data.get('keys_S', []))
        expected_vision_tokens = min(total_vision_tokens, self.sliding_window_size)

        init_token_len = 0
        if data.get('init_tokens') and data['init_tokens'][0] is not None:
            init_token_len = data['init_tokens'][0].shape[2]

        expected_total_tokens = init_token_len + expected_vision_tokens

        assert k_final.shape[2] == expected_total_tokens, (
            f"OQM Windowed KV Reconstruction Failed: Final token count mismatch. "
            f"Expected {expected_total_tokens} (init: {init_token_len} + vision: {expected_vision_tokens}), "
            f"but got {k_final.shape[2]}."
        )
        assert v_final.shape[2] == expected_total_tokens, "OQM Windowed KV Reconstruction Failed: V-cache token count mismatch."

        k_final = k_final.to(target_dtype)
        v_final = v_final.to(target_dtype)
        return k_final, v_final

import torch
from typing import Tuple


class StreamingTOMAttentionProcessor:

    def __init__(self, streamingtom_context, oqm):
        self.streamingtom_ctx = streamingtom_context
        self.oqm = oqm

    def process(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor,
                value_states: torch.Tensor, config) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.streamingtom_ctx.mode is None:
            return key_states, value_states

        assert self.streamingtom_ctx.video_id is not None, f"Layer {layer_idx}: streamingtom_ctx.video_id is None in {self.streamingtom_ctx.mode} mode!"
        assert self.streamingtom_ctx.mode in ['encode', 'retrieve'], f"Layer {layer_idx}: Invalid mode {self.streamingtom_ctx.mode}"

        if self.streamingtom_ctx.mode == 'encode' and self.streamingtom_ctx.should_store_keys:
            self._encode_mode(layer_idx, key_states, config)
        elif self.streamingtom_ctx.mode == 'retrieve':
            assert layer_idx not in self.streamingtom_ctx.retrieved_layers, \
                f"Layer {layer_idx}: Already retrieved"
            self.streamingtom_ctx.retrieved_layers.add(layer_idx)
            return self._retrieve_mode(layer_idx, query_states)

        return key_states, value_states

    def _encode_mode(self, layer_idx: int, key_states: torch.Tensor, config):
        batch, num_heads_kv, seq_len, head_dim = key_states.shape
        num_heads = config.num_attention_heads
        num_group = num_heads // num_heads_kv

        _ = self.streamingtom_ctx.batch_idx
        video_id = self.streamingtom_ctx.video_id
        _ = self.oqm.stored_tokens_count.get(video_id, {}).get(layer_idx, 0)

        self.streamingtom_ctx._encode_num_heads_kv = num_heads_kv
        self.streamingtom_ctx._encode_num_heads = num_heads
        self.streamingtom_ctx._encode_num_group = num_group

        token_keys = key_states.squeeze(0).transpose(0, 1).reshape(seq_len, num_heads_kv * head_dim)

        existing_group_keys = self.oqm.get_group_keys(self.streamingtom_ctx.video_id, layer_idx)
        if existing_group_keys is not None:
            assert existing_group_keys.shape[1] == num_heads_kv * head_dim, \
                f"Layer {layer_idx}: Dimension mismatch in incremental encoding"

        assert seq_len % self.oqm.group_size == 0, \
            f"Layer {layer_idx} Batch {self.streamingtom_ctx.batch_idx}: seq_len {seq_len} must be divisible by group_size {self.oqm.group_size}"

        self.oqm.store_token_keys_as_groups(self.streamingtom_ctx.video_id, layer_idx, token_keys)

    def _retrieve_mode(self, layer_idx: int, query_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        video_id = self.streamingtom_ctx.video_id
        budget = self.streamingtom_ctx.retrieval_info.get('budget', self.oqm.retrieval_max_tokens)

        group_keys = self.oqm.get_group_keys(video_id, layer_idx)
        assert group_keys is not None, (
            f"Missing group_keys for video {video_id} layer {layer_idx}. "
            f"Ensure video was properly encoded before retrieval."
        )

        assert query_states.shape[0] == 1, f"Expected batch_size=1, got {query_states.shape[0]}"
        _, num_heads_q, seq_q, head_dim = query_states.shape

        num_heads_kv = self.streamingtom_ctx._encode_num_heads_kv if hasattr(self.streamingtom_ctx, '_encode_num_heads_kv') else num_heads_q
        num_group = num_heads_q // num_heads_kv

        if num_group > 1:
            query_states_grouped = query_states.reshape(1, num_heads_kv, num_group, seq_q, head_dim)
            query_states_kv = query_states_grouped.mean(dim=2)
        else:
            query_states_kv = query_states

        query_repr = query_states_kv[0].mean(dim=1).reshape(-1)

        assert query_repr.numel() == group_keys.shape[1], \
            f"Layer {layer_idx}: Query dim {query_repr.numel()} != group_keys dim {group_keys.shape[1]}"

        self.streamingtom_ctx.current_layer = layer_idx

        selected_indices = self._select_top_k_groups(query_repr, group_keys, budget)
        self.streamingtom_ctx.selected_vision_group_indices_per_layer[layer_idx] = selected_indices

        layer_kv = self.oqm.get_selective_kv(video_id, layer_idx, selected_indices)
        assert layer_kv is not None, (
            f"Failed to reconstruct KV for video {video_id} layer {layer_idx}. "
            f"Selected groups: {selected_indices.tolist()}"
        )

        retrieved_k, retrieved_v = layer_kv
        expected = self.oqm.init_token_count + len(selected_indices) * self.oqm.group_size
        actual = retrieved_k.shape[2]
        assert actual == expected, (
            f"Token count mismatch in layer {layer_idx}: "
            f"expected {expected}, got {actual}"
        )

        return retrieved_k, retrieved_v

    def _select_top_k_groups(self, query_repr: torch.Tensor, group_keys: torch.Tensor, budget: int) -> torch.Tensor:
        assert query_repr.dim() == 1 and group_keys.dim() == 2
        assert query_repr.shape[0] == group_keys.shape[1]
        assert budget > 0

        tokens_per_group = self.oqm.group_size
        num_groups = group_keys.shape[0]
        budget_groups = (budget + tokens_per_group - 1) // tokens_per_group
        assert num_groups > 0 and budget_groups > 0

        if budget_groups >= num_groups:
            return torch.arange(num_groups, device=group_keys.device)
        if budget_groups <= 0:
            return torch.tensor([], device=group_keys.device, dtype=torch.long)

        query_repr_norm = torch.nn.functional.normalize(query_repr.float(), p=2, dim=0)
        group_keys_norm = torch.nn.functional.normalize(group_keys.float(), p=2, dim=1)
        similarities = torch.matmul(group_keys_norm, query_repr_norm)
        assert not torch.isnan(similarities).any() and not torch.isinf(similarities).any()

        _, top_indices = torch.topk(similarities, k=min(budget_groups, num_groups), sorted=False)
        assert len(top_indices) > 0 and len(top_indices) <= budget_groups

        sorted_indices = torch.sort(top_indices)[0]
        return sorted_indices

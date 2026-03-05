import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from ..main import get_streamingtom_config

class CTR:

    def __init__(self):
        config = get_streamingtom_config().config
        self.retain_tokens = config['ctr_retain_tokens']
        self.similarity_threshold = config['ctr_similarity_threshold']
        self.k = config['ctr_k']
        self.beta = config['ctr_beta']

    def _select_tokens_by_attention(self, features: torch.Tensor, attention_scores: torch.Tensor,
                                    mask: torch.Tensor, num_to_keep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_to_keep <= 0 or not mask.any():
            device = features.device
            return (torch.zeros(0, features.shape[-1], device=device, dtype=features.dtype),
                    torch.tensor([], device=device, dtype=torch.long))

        masked_scores = attention_scores[mask]
        k = min(num_to_keep, len(masked_scores))
        if k == 0:
            device = features.device
            return (torch.zeros(0, features.shape[-1], device=device, dtype=features.dtype),
                    torch.tensor([], device=device, dtype=torch.long))

        _, top_indices = torch.topk(masked_scores, k)
        positions = torch.where(mask)[0]
        selected_positions = positions[top_indices]
        return features[selected_positions], selected_positions

    def _select_and_merge_tokens_by_dpc(self, features: torch.Tensor, mask: torch.Tensor,
                                        num_clusters: int, k: int, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_clusters <= 0 or not mask.any():
            device = features.device
            return (torch.zeros(0, features.shape[-1], device=device, dtype=features.dtype),
                    torch.tensor([], device=device, dtype=torch.long))

        masked_features = features[mask]
        seq_len, embed_dim = masked_features.shape
        num_clusters = min(num_clusters, seq_len)
        if num_clusters <= 0:
            device = features.device
            return (torch.zeros(0, features.shape[-1], device=device, dtype=features.dtype),
                    torch.tensor([], device=device, dtype=torch.long))

        centroid_indices = self._dpc_cluster(masked_features, num_clusters, k)
        positions = torch.where(mask)[0]
        selected_positions = positions[centroid_indices]

        if num_clusters == seq_len:
            return masked_features[centroid_indices], selected_positions

        with torch.no_grad():
            dist_matrix = torch.cdist(masked_features.float(), masked_features.float()) / (embed_dim ** 0.5)

        all_indices = torch.arange(seq_len, device=features.device)
        non_center_mask = ~torch.isin(all_indices, centroid_indices)
        non_center_indices = all_indices[non_center_mask]
        if non_center_indices.numel() == 0:
            return masked_features[centroid_indices], selected_positions

        non_center_features = masked_features[non_center_indices]
        dist_to_centers = dist_matrix[non_center_indices][:, centroid_indices]
        assignments = torch.argmin(dist_to_centers, dim=1)

        merged_features = []
        for i, center_idx in enumerate(centroid_indices):
            cluster_mask = (assignments == i)
            if cluster_mask.any():
                cluster_mean = non_center_features[cluster_mask].mean(dim=0)
                merged = beta * masked_features[center_idx] + (1 - beta) * cluster_mean
            else:
                merged = masked_features[center_idx]
            merged_features.append(merged)

        return torch.stack(merged_features), selected_positions

    def _dpc_cluster(self, tokens: torch.Tensor, cluster_num: int, k: int) -> torch.Tensor:
        with torch.no_grad():
            seq_len, embed_dim = tokens.shape
            if seq_len <= 1:
                return torch.arange(seq_len, device=tokens.device)

            k_neighbors = min(k, seq_len - 1)
            n_clusters = min(cluster_num, seq_len)

            dist_matrix = torch.cdist(tokens.float(), tokens.float()) / (embed_dim ** 0.5)

            nearest_dists, _ = torch.topk(dist_matrix, k_neighbors, dim=-1, largest=False)
            density = (-(nearest_dists ** 2).mean(dim=-1)).exp()
            density += torch.rand_like(density) * 1e-6

            higher_density_mask = density[:, None] > density[None, :]
            dist_to_higher = torch.where(higher_density_mask, dist_matrix, dist_matrix.max()).min(dim=-1)[0]

            scores = density * dist_to_higher
            _, centers = torch.topk(scores, n_clusters)
            return torch.sort(centers)[0]

    def _classify_tokens(self, vision_tokens_batch: torch.Tensor, ctr_state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        num_frames, num_tokens = vision_tokens_batch.shape[:2]
        static_masks = torch.zeros((num_frames, num_tokens), dtype=torch.bool, device=vision_tokens_batch.device)
        dynamic_masks = torch.zeros_like(static_masks)
        last_frame = ctr_state.get('last_frame_tokens')

        for i in range(num_frames):
            curr_frame = vision_tokens_batch[i]

            if i == 0 and last_frame is None:
                dynamic_masks[i] = True
            else:
                prev_frame = last_frame[0] if i == 0 else vision_tokens_batch[i-1]
                curr_norm = F.normalize(curr_frame.float(), p=2, dim=-1)
                prev_norm = F.normalize(prev_frame.float(), p=2, dim=-1)
                similarity = (curr_norm * prev_norm).sum(dim=-1)
                static_masks[i] = similarity > self.similarity_threshold
                dynamic_masks[i] = ~static_masks[i]

        ctr_state['last_frame_tokens'] = vision_tokens_batch[-1:].detach()
        return static_masks, dynamic_masks

    def _compress_single_frame(self, frame_idx: int, vision_features_batch: torch.Tensor,
                               attention_scores_batch: torch.Tensor,
                               static_masks: torch.Tensor, dynamic_masks: torch.Tensor) -> torch.Tensor:
        frame_features = vision_features_batch[frame_idx]
        frame_scores = attention_scores_batch[frame_idx]
        static_mask = static_masks[frame_idx]
        dynamic_mask = dynamic_masks[frame_idx]

        assert frame_features.shape[0] == 196, f"Frame {frame_idx}: Expected 196 tokens but got {frame_features.shape[0]}"

        static_count = static_mask.sum().item()
        dynamic_count = dynamic_mask.sum().item()
        total_count = static_count + dynamic_count

        assert total_count == 196, f"Frame {frame_idx}: Classification error! Static+Dynamic={total_count}, expected 196"

        static_budget = int(self.retain_tokens * static_count / total_count)
        dynamic_budget = self.retain_tokens - static_budget

        if static_budget > static_count:
            static_budget = static_count
            dynamic_budget = self.retain_tokens - static_budget
        if dynamic_budget > dynamic_count:
            dynamic_budget = dynamic_count
            static_budget = self.retain_tokens - dynamic_budget

        collected_features, collected_positions = [], []

        if static_budget > 0:
            feats, pos = self._select_and_merge_tokens_by_dpc(
                frame_features, static_mask, static_budget, self.k, self.beta)
            if feats.numel() > 0:
                collected_features.append(feats)
                collected_positions.append(pos)

        if dynamic_budget > 0:
            feats, pos = self._select_tokens_by_attention(
                frame_features, frame_scores, dynamic_mask, dynamic_budget)
            if feats.numel() > 0:
                collected_features.append(feats)
                collected_positions.append(pos)

        assert collected_features, f"Frame {frame_idx}: Failed to collect any features!"

        all_positions = torch.cat(collected_positions)
        all_features = torch.cat(collected_features, dim=0)
        result = all_features[torch.argsort(all_positions)]

        if result.shape[0] > self.retain_tokens:
            result = result[:self.retain_tokens]

        assert result.shape[0] == self.retain_tokens, \
            f"Frame {frame_idx}: Expected {self.retain_tokens} tokens but got {result.shape[0]}"

        return result

    def compress_features(self, vision_features_batch: torch.Tensor, current_state: Dict[str, Any],
                         attention_scores_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        ctr_state = current_state.get('ctr_state', {})

        num_frames = vision_features_batch.shape[0]
        assert num_frames > 0, "No frames to compress"

        static_masks, dynamic_masks = self._classify_tokens(vision_features_batch, ctr_state)

        compressed_frames = []
        for i in range(num_frames):
            compressed = self._compress_single_frame(
                i, vision_features_batch, attention_scores_batch,
                static_masks, dynamic_masks
            )
            compressed_frames.append(compressed)

        compressed_result = torch.stack(compressed_frames).reshape(-1, vision_features_batch.shape[-1])

        return compressed_result, {'ctr_state': ctr_state}

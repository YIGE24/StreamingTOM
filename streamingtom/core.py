from typing import Any, Dict, Optional, Tuple

import torch

from .modules.ctr import CTR
from .modules.oqm import OQM
from .tasks.vision import VisionTask
from .tasks.query import QueryTask
from .main import get_streamingtom_config

class StreamingTOMCore:

    def __init__(self):
        config = get_streamingtom_config().config
        self.oqm = OQM()
        self.vision_task = VisionTask(CTR(), self.oqm)
        self.query_task = QueryTask(self.oqm, config)

    def encode_vision_batch(self, frame_batch: Dict, vision_tower, mm_projector=None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vision_task.encode_vision_batch(frame_batch, vision_tower, mm_projector)

    def process_vision_batch(self, video_id: str, processed_features: torch.Tensor, current_state: Dict[str, Any],
                           attention_scores, batch_idx: int, model_forward_fn, frame_info: Dict[str, Any]) -> Dict[str, Any]:
        assert frame_info is not None
        return self.vision_task.process_vision_batch(
            video_id, processed_features, current_state, attention_scores,
            batch_idx, model_forward_fn, frame_info
        )

    def prepare_retrieval_context(self, video_id: str, input_ids, model_forward_fn, tokenizer):
        return self.query_task.prepare_retrieval_context(
            video_id, input_ids, model_forward_fn, tokenizer
        )

    def retrieve_and_generate(self, input_ids, model_forward_fn, tokenizer, **kwargs):
        return self.query_task.retrieve_and_generate(
            input_ids, model_forward_fn, tokenizer, **kwargs
        )

    def clear_cache(self, video_id: Optional[str] = None):
        self.oqm.clear_cache(video_id)

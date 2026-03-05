from typing import Any, Dict, Tuple

import torch


class StreamingTOMPipeline:

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.model_forward = self.model.forward
        self.model_type = getattr(self.model, '_streamingtom_model_type')
        assert self.model_type == 'llava', f"Unsupported model type {self.model_type}, only 'llava' is supported"
        self.streamingtom_core = getattr(self.model, '_streamingtom_core_obj')
        self.config = getattr(self.model, 'streamingtom_config').config
        self.streamingtom_state = {}
        self.streamingtom_core.clear_cache()

    def process_video(self, video_id: str, frame_batches: list, questions: list = None, **kwargs):
        encoding_start_frame = kwargs.pop('encoding_start_frame', None)

        batch_info = self.streamingtom_core.oqm.get_batch_info(video_id)
        cumulative_batch_offset = batch_info.get('batch_idx', 0)

        total_batches = len(frame_batches)

        for local_batch_idx, frame_batch in enumerate(frame_batches):
            batch_idx = local_batch_idx + cumulative_batch_offset
            is_first = (batch_idx == 0)

            frame_info = {
                'batch_type': {
                    'is_first': is_first,
                    'is_last': local_batch_idx == total_batches - 1,
                    'batch_idx': batch_idx
                },
                'local_batch_idx': local_batch_idx,
                'encoding_start_frame': encoding_start_frame
            }

            frame_batch['frame_info'] = frame_info
            self._process_single_batch(video_id, batch_idx, frame_batch, frame_info)

        if questions is None:
            return None

        answers = []
        for question in questions:
            q_kwargs = {k: v for k, v in question.items() if k not in ['batch_idx', 'input_ids']}
            q_kwargs.update(kwargs)
            answers.append({
                'answer': self._process_query(video_id, question['input_ids'], **q_kwargs)
            })

        return answers

    def reset_streamingtom_state(self):
        self.streamingtom_state = {}
        self.streamingtom_core.clear_cache()

    def _process_single_batch(self, video_id: str, batch_idx: int, frame_batch: Dict, frame_info: Dict[str, Any]):
        processed_features, attention_scores = self._encode_frame_batch(frame_batch)
        self.streamingtom_state = self.streamingtom_core.process_vision_batch(
            video_id,
            processed_features,
            self.streamingtom_state,
            attention_scores,
            batch_idx,
            self.model_forward,
            frame_info
        )

    def _get_vision_tower(self):
        if hasattr(self.model, 'get_vision_tower'):
            return self.model.get_vision_tower()
        if hasattr(self.model, 'visual'):
            return self.model.visual
        return None

    def _encode_frame_batch(self, frame_batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        vision_tower = self._get_vision_tower()
        mm_projector = None

        if self.model_type == "llava":
            model_instance = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
            mm_projector = getattr(model_instance, 'mm_projector', None)

        return self.streamingtom_core.encode_vision_batch(
            frame_batch,
            vision_tower,
            mm_projector
        )

    def _process_query(self, video_id: str, input_ids, **kwargs):
        self.streamingtom_core.prepare_retrieval_context(
            video_id,
            input_ids,
            self.model_forward,
            self.tokenizer
        )
        return self.streamingtom_core.retrieve_and_generate(
            input_ids,
            self.model_forward,
            self.tokenizer,
            **kwargs
        )

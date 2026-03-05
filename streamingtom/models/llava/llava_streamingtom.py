import os
import types
import uuid
from typing import Dict, Iterator

import torch
import torch.nn as nn

from ...core import StreamingTOMCore
from ...pipeline import StreamingTOMPipeline
from .modeling_qwen2_revise import Qwen2Attention

def setup_llava_with_streamingtom(model: nn.Module, streamingtom_config) -> nn.Module:
    model.streamingtom_config = streamingtom_config
    model._streamingtom_core_obj = StreamingTOMCore()
    _replace_attention_layers(model)
    model.original_generate = model.generate
    model.generate = types.MethodType(generate_with_streamingtom, model)
    model.generate_with_streamingtom_streaming = types.MethodType(generate_with_streamingtom_streaming, model)
    model._is_streamingtom_patched = True
    return model

def create_frame_generator_llava(video_input, config) -> Iterator[Dict]:
    assert isinstance(video_input, torch.Tensor), f"Expected tensor input, got {type(video_input)}"

    if video_input.shape[0] == 0:
        return

    assert video_input.dim() == 4, f"Expected 4D tensor [T, C, H, W], got {video_input.dim()}D"
    batch_size = config.get('streaming_encoder_batch_size')
    total_frames = video_input.shape[0]

    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        yield {'frames': list(video_input[start:end].unbind(0)), 'grid_thw': None}

def generate_with_streamingtom(self, *args, **kwargs):
    input_ids = args[0]
    images = kwargs.get('images')

    if images is None:
        return self.original_generate(*args, **kwargs)
    images = images[0]

    if not hasattr(self, '_streamingtom_pipeline'):
        tokenizer = self.tokenizer if hasattr(self, 'tokenizer') else kwargs.get('tokenizer')
        assert tokenizer is not None, "tokenizer is required for StreamingTOM inference"
        self._streamingtom_pipeline = StreamingTOMPipeline(model=self, tokenizer=tokenizer)
    else:
        self._streamingtom_pipeline.reset_streamingtom_state()

    video_id = f"video_{uuid.uuid4().hex[:8]}"
    frame_batches = list(create_frame_generator_llava(images, self.streamingtom_config))
    questions = [{'batch_idx': -1, 'input_ids': input_ids}]
    answers = self._streamingtom_pipeline.process_video(video_id, frame_batches, questions, **kwargs)
    self._streamingtom_pipeline.streamingtom_core.clear_cache(video_id)

    if os.getenv('STREAMINGTOM_PROFILE', '0') == '1':
        from ...utils.profiler import get_profiler
        profiler = get_profiler()
        profiler.print_summary()

    return answers[0]['answer'] if answers else None

def generate_with_streamingtom_streaming(self, images, questions, video_id=None, is_start=False, **kwargs):
    assert video_id is not None, "video_id must be provided"

    if not hasattr(self, '_video_pipelines'):
        self._video_pipelines = {}

    needs_new_pipeline = is_start or video_id not in self._video_pipelines

    if needs_new_pipeline:
        if is_start:
            for vid, pipeline in list(self._video_pipelines.items()):
                pipeline.streamingtom_core.clear_cache(vid)
            self._video_pipelines.clear()

        tokenizer = getattr(self, 'tokenizer', None) or kwargs.get('tokenizer')
        assert tokenizer is not None, "tokenizer is required for StreamingTOM inference"
        self._video_pipelines[video_id] = StreamingTOMPipeline(model=self, tokenizer=tokenizer)

    pipeline = self._video_pipelines[video_id]

    if images is not None and images.numel() > 0:
        frame_batches = list(create_frame_generator_llava(images, self.streamingtom_config))
    else:
        frame_batches = []

    answers = pipeline.process_video(video_id, frame_batches, questions, **kwargs)
    return answers

def _replace_attention_layers(model: nn.Module):
    if not (hasattr(model, 'model') and hasattr(model.model, 'layers')):
        raise ValueError(f"Cannot find model.model.layers, model type: {type(model)}")

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, 'self_attn'):
            continue
        old_attn = layer.self_attn
        new_attn = Qwen2Attention(config=old_attn.config, layer_idx=layer_idx).to(
            dtype=old_attn.q_proj.weight.dtype, device=old_attn.q_proj.weight.device)
        new_attn.load_state_dict(old_attn.state_dict())
        layer.self_attn = new_attn
        layer.self_attn._streamingtom_context = None

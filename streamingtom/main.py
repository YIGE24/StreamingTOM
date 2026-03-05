import os
from typing import Any, Dict


class StreamingTOMConfig:

    def __init__(self):
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        config = {
            'ctr_similarity_threshold': float(os.environ['CTR_SIMILARITY_THRESHOLD']),
            'ctr_retain_tokens': int(os.environ['CTR_RETAIN_TOKENS']),
            'ctr_k': int(os.environ['CTR_K']),
            'ctr_beta': float(os.environ['CTR_BETA']),
            'oqm_retrieval_max_tokens': int(os.environ['OQM_RETRIEVAL_MAX_TOKENS']),
            'oqm_enable_quantization': os.environ['OQM_ENABLE_QUANTIZATION'] == '1',
            'oqm_quantization_bits': int(os.environ['OQM_QUANTIZATION_BITS']),
            'oqm_group_size': int(os.environ['OQM_GROUP_SIZE']),
            'oqm_init_token_count': int(os.environ['OQM_INIT_TOKEN_COUNT']),
            'oqm_sliding_window_size': int(os.environ['OQM_SLIDING_WINDOW_SIZE']),
            'streaming_encoder_batch_size': int(os.environ['STREAMING_ENCODER_BATCH_SIZE']),
        }
        return config

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def _validate_config(self):
        assert self.config['ctr_retain_tokens'] == self.config['oqm_group_size'], \
            "ctr_retain_tokens must equal oqm_group_size"
        assert self.config['ctr_retain_tokens'] > 0, "ctr_retain_tokens must be > 0"
        assert 0 <= self.config['ctr_similarity_threshold'] <= 1, "ctr_similarity_threshold must be in [0, 1]"
        assert self.config['ctr_k'] > 0, "ctr_k must be > 0"
        assert 0 <= self.config['ctr_beta'] <= 1, "ctr_beta must be in [0, 1]"
        assert self.config['oqm_retrieval_max_tokens'] > 0, "oqm_retrieval_max_tokens must be > 0"
        assert isinstance(self.config['oqm_enable_quantization'], bool), "oqm_enable_quantization must be bool"
        assert self.config['oqm_quantization_bits'] in [2, 4], "oqm_quantization_bits must be 2 or 4"
        assert self.config['oqm_group_size'] > 0, "oqm_group_size must be > 0"
        assert self.config['oqm_init_token_count'] >= 0, "oqm_init_token_count must be >= 0"
        assert self.config['oqm_sliding_window_size'] > 0, "oqm_sliding_window_size must be > 0"
        assert self.config['streaming_encoder_batch_size'] > 0, "streaming_encoder_batch_size must be > 0"
        assert self.config['oqm_sliding_window_size'] % self.config['oqm_group_size'] == 0, \
            "oqm_sliding_window_size must be a multiple of oqm_group_size"


_STREAMINGTOM_CONFIG_INSTANCE = None


def streamingtom(model, model_type: str):
    from .models.llava.llava_streamingtom import setup_llava_with_streamingtom
    config = reload_streamingtom_config()
    _print_initialization_info(model_type, config)
    model_type = model_type.lower()
    if model_type == "llava":
        return _patch_model(model, config, setup_llava_with_streamingtom, model_type)
    raise ValueError(f"Unsupported model type {model_type}, only 'llava' is supported")


def get_streamingtom_config():
    global _STREAMINGTOM_CONFIG_INSTANCE
    if _STREAMINGTOM_CONFIG_INSTANCE is None:
        _STREAMINGTOM_CONFIG_INSTANCE = StreamingTOMConfig()
    return _STREAMINGTOM_CONFIG_INSTANCE


def reload_streamingtom_config():
    global _STREAMINGTOM_CONFIG_INSTANCE
    _STREAMINGTOM_CONFIG_INSTANCE = StreamingTOMConfig()
    return _STREAMINGTOM_CONFIG_INSTANCE


def _patch_model(model, config, setup_func, model_type: str):
    if getattr(model, '_is_streamingtom_patched', False):
        return model
    model._streamingtom_model_type = model_type
    setup_func(model, config)
    return model


def _print_initialization_info(model_type: str, config):
    print(f"\n{'='*60}")
    print(f"Model Type: {model_type}")
    for key, value in sorted(config.config.items()):
        print(f"  {key}: {value}")
    print("="*60 + "\n")

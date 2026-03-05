class StreamingTOMContext:
    _instance = None

    def __init__(self):
        self.mode = None
        self.video_id = None
        self.batch_idx = 0
        self.oqm = None
        self.retrieval_info = {}
        self.retrieved_layers = set()
        self.selected_vision_group_indices_per_layer = {}
        self.should_store_keys = False
        self.question_kv_per_layer = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_encode_mode(self, video_id, batch_idx=0):
        assert len(self.retrieved_layers) == 0, "Retrieved layers not cleared"
        assert len(self.selected_vision_group_indices_per_layer) == 0, "Selected indices not cleared"
        self.mode = 'encode'
        self.video_id = video_id
        self.batch_idx = batch_idx

    def clear_mode(self):
        self.mode = None
        self.video_id = None
        self.batch_idx = 0
        self.oqm = None
        self.retrieval_info = {}
        self.retrieved_layers.clear()
        self.selected_vision_group_indices_per_layer.clear()
        self.should_store_keys = False
        self.question_kv_per_layer.clear()

    def inject_to_model(self, model):
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            layers = model.language_model.layers

        assert layers is not None, f"Failed to get model layers for {type(model).__name__}"
        assert len(layers) == 28, f"Expected 28 layers, got {len(layers)}"

        for layer_idx, layer in enumerate(layers):
            assert hasattr(layer, 'self_attn'), f"Layer {layer_idx} has no self_attn"
            layer.self_attn._streamingtom_context = self

    def set_oqm(self, oqm):
        self.oqm = oqm

    @staticmethod
    def get_model_num_layers(model):
        config = getattr(model, 'config', None)
        if not config:
            raise AttributeError(f"Model {type(model).__name__} has no config")
        num_layers = getattr(getattr(config, 'text_config', None), 'num_hidden_layers', None) or \
                     getattr(config, 'num_hidden_layers', None)
        if num_layers is None:
            raise AttributeError(f"Cannot find num_hidden_layers for {type(model).__name__}")
        return num_layers

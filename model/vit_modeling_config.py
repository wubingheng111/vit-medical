from transformers.configuration_utils import PretrainedConfig


class VitmodelConfig(PretrainedConfig):
    model_type = "vitmodel"
    is_composition = True

    def __init__(self, num_labels=0, image_size=224, patch_size=16, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = "gelu"
        
import torch
import torch.nn as nn
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch.nn.init as init
from transformers import LlavaNextPreTrainedModel, LlavaNextConfig
from safetensors.torch import load_file
import os

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ValueModel(LlavaNextPreTrainedModel):
    def __init__(self, base_model_pth, hidden_size=2560 * 4096, output_size=1):
        config = LlavaNextConfig.from_pretrained(base_model_pth)
        super(ValueModel, self).__init__(config)
        self.base_model_pth = base_model_pth
        self.hidden_size = hidden_size
        self.flatten = Flatten()
        self.processor = LlavaNextProcessor.from_pretrained(self.base_model_pth)
        self.llava_encoder = LlavaNextForConditionalGeneration.from_pretrained(self.base_model_pth,
                                                                               torch_dtype=torch.float16)

        self.v_head = nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
        )

        self._initialize_weights(self.v_head)

    def forward(self, inputs, attention_mask=None):
        latent_output = self.llava_encoder(**inputs, output_hidden_states=True, max_embed_length=3000)
        hidden_states = self.flatten(latent_output.hidden_states[-1])
        output = self.v_head(hidden_states)
        return output

    def _initialize_weights(self, module):
        for layer in module:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def from_pretrained(self, pretrained_model_pth):
        complete_state_dict = {}
        for i in range(4):
            complete_state_dict.update(
                load_file(os.path.join(pretrained_model_pth, "model-0000{}-of-00004.safetensors".format(i + 1))))

        llava_encoder_dict = {k.replace('llava_encoder.', ''): v for k, v in complete_state_dict.items()}
        value_keys = ["v_head.0.bias", "v_head.0.weight"]
        value_dict = {}
        for key in value_keys:
            if key in complete_state_dict:
                value_dict[key] = complete_state_dict[key]
                del llava_encoder_dict[key]

        value_dict = {k.replace('v_head.', ''): v for k, v in value_dict.items()}
        self.llava_encoder.load_state_dict(llava_encoder_dict)
        self.v_head.load_state_dict(value_dict)

import json
import torch
from trl import (
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    TDTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from vlm_value_models import ValueModel
from PIL import Image


# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def load_value_dataset(dataset_pth):
    datas = []
    with open(dataset_pth, 'r') as files:
        for line in files:
            datas.append(json.loads(line))

    td_pairs = []
    for data in datas:
        for key in data['rewards'].keys():
            for i in range(len(list(data['rewards'][key].keys())) - 1):
                td_pairs.append({
                    'image': data['image_path'],
                    'current_state': list(data['rewards'][key].keys())[i],
                    'next_state': list(data['rewards'][key].keys())[i + 1],
                    'reward': data['rewards'][key][list(data['rewards'][key].keys())[i]]
                })

    return td_pairs

def load_td_dataset(td_pth):
    datas = []
    with open(td_pth, 'r') as files:
        for line in files:
            datas.append(json.loads(line))

    return datas

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.dataset_text_field = ""  # need a dummy field
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model_pth = 'llava-hf/llava-v1.6-mistral-7b-hf'
    value_net = ValueModel(model_pth)

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        current_state = [value_net.processor.apply_chat_template([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example["current_state"]},
                        {"type": "image"},
                    ],
                },
            ], tokenize=False) for example in examples]
        next_state = [value_net.processor.apply_chat_template([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example["next_state"]},
                        {"type": "image"},
                    ],
                },
            ], tokenize=False) for example in examples]
        images = [Image.open(example["image"]) for example in examples]
        rewards = [example["reward"] for example in examples]

        # Tokenize the texts and process the images
        batch = value_net.processor(text=current_state, images=images, padding='max_length', max_length=2560, truncation=True, return_tensors="pt")
        batch_next = value_net.processor(text=next_state, images=images, padding='max_length', max_length=2560, truncation=True, return_tensors="pt")
        batch['next_input_ids'] = batch_next['input_ids'].clone()
        batch['next_attention_mask'] = batch_next['attention_mask'].clone()
        batch['next_pixel_values'] = batch_next['pixel_values'].clone()
        batch['next_image_sizes'] = batch_next['image_sizes'].clone()
        batch["rewards"] = torch.tensor(rewards)


        return batch

    ################
    # Dataset
    ################
    td_dataset_pth = './filtered_data.jsonl' # Your TD dataset path here
    td_dataset = load_td_dataset(td_dataset_pth)

    ################
    # Training
    ################
    trainer = TDTrainer(
        model=value_net,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=td_dataset,
        tokenizer=value_net.processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
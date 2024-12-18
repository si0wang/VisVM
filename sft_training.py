import json
import torch
from trl import (
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, GenerationConfig
from PIL import Image

def load_dataset(dataset_pth):
    datas = []
    with open(dataset_pth, 'r') as files:
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
    processor = LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')
    model = LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf', torch_dtype=torch.float16,
                                                              low_cpu_mem_usage=True)

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        text_inputs = [processor.apply_chat_template([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example["text"]},
                        {"type": "image"},
                    ],
                },
            {
                "role": "assistant",
                "content": [
                            {"type": "text", "text": example["decoding_result"]},
                ],
            },
            ], tokenize=False) for example in examples]
        images = [Image.open(example["image"]) for example in examples]

        batch = processor(text=text_inputs, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    dataset_pth = '' # Your control decoding caption file path
    datasets = load_dataset(dataset_pth)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=datasets,
        tokenizer=processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
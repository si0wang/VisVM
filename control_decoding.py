from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, GenerationConfig
import torch
from PIL import Image
import requests
import json
from vlm_value_models import ValueModel
import math
from tqdm import tqdm
import argparse
import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_data(data_path):
    datas = []
    with open(data_path, 'r') as files:
        for line in files:
            datas.append(json.loads(line))
    return datas

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def main(args):
    device = "cuda:{}".format(args.gpu_id)
    ############# Load data ##############
    datas = load_data(args.data_pth)
    data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)

    ############# Load VLM ##############
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device)

    ############# Load Value net ##############
    value_net = ValueModel(args.model_id)
    value_net.to(device, dtype=torch.float16)
    value_net.from_pretrained(args.value_net_pth)

    decoding_results = []
    for data in tqdm(data_chunk, desc="Decoding Progress"):
        try:
            images = [Image.open(data['image_path'])]
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": data['text']},
                    {"type": "image"},
                ],
            }]

            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=images, text=[prompt], return_tensors="pt").to(device)
            question_input_length = inputs['input_ids'].shape[1]
            temp_generation_config_list = [
                GenerationConfig(
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    do_sample=False,
                )
            ]

            max_value = -99999
            for temp_generation_config in temp_generation_config_list:
                for i in np.arange(args.step_size):
                    with torch.no_grad():
                        output = model.generate(**inputs, generation_config=temp_generation_config, max_length=4096,
                                                stop_strings=['.'], tokenizer=processor.tokenizer)

                    new_generated_reply = processor.decode(output[0][question_input_length:], skip_special_tokens=True)
                    current_state = [value_net.processor.apply_chat_template([
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": new_generated_reply},
                                {"type": "image"},
                            ],
                        },
                    ], tokenize=False)]
                    batch = value_net.processor(text=current_state, images=images, padding='max_length', max_length=2560,
                                                truncation=True, return_tensors="pt").to(device)
                    current_inputs = {'input_ids': batch['input_ids'],
                                      'attention_mask': batch['attention_mask'],
                                      'pixel_values': batch['pixel_values'],
                                      'image_sizes': batch['image_sizes'], }

                    with torch.no_grad():
                        current_value = value_net(current_inputs)
                        del batch
                    if current_value > max_value:
                        max_value = current_value
                        chosen_response = new_generated_reply
            del inputs
            assistant_reply = None

            while assistant_reply != new_generated_reply:
                max_value = -99999
                assistant_reply = new_generated_reply
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": data['text']},
                            {"type": "image"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": '{TEXT}'}, ],
                    }]
                conversation[-1]['content'][0]['text'] = chosen_response
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                prompt = prompt[:-5]  # remove </s>
                inputs = processor(images=images, text=[prompt], return_tensors="pt").to(device)

                reply_input_length = inputs['input_ids'].shape[1]

                for temp_generation_config in temp_generation_config_list:
                    for i in np.arange(args.step_size):
                        with torch.no_grad():
                            output = model.generate(**inputs, generation_config=temp_generation_config, max_length=4096,
                                                    stop_strings=['.'], tokenizer=processor.tokenizer)

                        reply_candidate = processor.decode(output[0][question_input_length:], skip_special_tokens=True)
                        new_generated_reply = processor.decode(output[0][reply_input_length:], skip_special_tokens=True)
                        current_state = [value_net.processor.apply_chat_template([
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": new_generated_reply},
                                    {"type": "image"},
                                ],
                            },
                        ], tokenize=False)]
                        batch = value_net.processor(text=current_state, images=images, padding='max_length', max_length=2560,
                                                    truncation=True, return_tensors="pt").to(device)
                        current_inputs = {'input_ids': batch['input_ids'],
                                          'attention_mask': batch['attention_mask'],
                                          'pixel_values': batch['pixel_values'],
                                          'image_sizes': batch['image_sizes'], }

                        with torch.no_grad():
                            current_value = value_net(current_inputs)
                            del batch
                        if current_value > max_value:
                            max_value = current_value
                            chosen_response = reply_candidate
                del inputs

            decoding_results.append({
                    'text': data['text'],
                    'image': data['image'],
                    'image_path': data['image_path'],
                    'decoding_result': chosen_response,
                })
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    dump_to_jsonl(decoding_results, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--data_pth", type=str, default=None)
    parser.add_argument("--value_net_pth", type=str, default=None)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--per_gpu_batch_size", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    main(args)
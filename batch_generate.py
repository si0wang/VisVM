import os
# os.environ['http_proxy'] = 'http://172.18.166.31:7899'
# os.environ['https_proxy'] = os.environ['http_proxy']

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, GenerationConfig
from tqdm import tqdm
import json
import math
import argparse
from PIL import Image as Img

def get_llava_dataset(llava_data_pth, llava_image_pth):
    llava_datas = []

    with open(llava_data_pth, 'r', encoding='utf-8') as file:
        for line in file:
            llava_datas.append(json.loads(line))
    datas = []
    for data in llava_datas:
        prompt = data['prompt']
        if prompt.startswith("<image>\n"):
            prompt = data['prompt'][len("<image>\n"):]
        if prompt.endswith("\n<image>"):
            prompt = data['prompt'][:-len("\n<image>")]
        datas.append({'text': prompt,
                      'image': '{}/{}'.format(llava_image_pth, data['image']),
                      'image_path': '{}/{}'.format(llava_image_pth, data['image'])})

    return datas

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def main(args):
    device = "cuda:{}".format(args.gpu_id)
    dataset = get_llava_dataset(llava_data_pth=args.llava_data_pth, llava_image_pth=args.image_folder)
    data_chunk = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    temp_generation_config_1 = GenerationConfig(
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        max_new_tokens=1024,
        min_length=1
    )

    temp_generation_config_2 = GenerationConfig(
        temperature=0.4,
        do_sample=True,
        top_p=0.9,
        max_new_tokens=1024,
        min_length=1
    )

    temp_generation_config_3 = GenerationConfig(
        temperature=0.6,
        do_sample=True,
        top_p=0.9,
        max_new_tokens=1024,
        min_length=1
    )

    temp_generation_config_4 = GenerationConfig(
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        max_new_tokens=1024,
        min_length=1
    )

    greedy_generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=1024
    )

    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device)

    final_response = []
    for i in tqdm(range(0, len(data_chunk), args.per_gpu_batch_size), desc="Batch Inference Progress"):
        if i + args.per_gpu_batch_size < len(data_chunk):
            batch_queries = data_chunk[i:i + args.per_gpu_batch_size]
        else:
            batch_queries = data_chunk[i:]
        images, prompts = [], []
        for data in batch_queries:
            images.append(Img.open(data['image_path']))
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": data['text']},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            prompts.append(prompt)
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)

        temperature_output_1 = model.generate(**inputs, generation_config=temp_generation_config_1, tokenizer=processor.tokenizer, max_new_tokens=2048)
        temperature_decodes_1 = processor.batch_decode(temperature_output_1, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)

        temperature_output_2 = model.generate(**inputs, generation_config=temp_generation_config_2, tokenizer=processor.tokenizer, max_new_tokens=2048)
        temperature_decodes_2 = processor.batch_decode(temperature_output_2, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)

        temperature_output_3 = model.generate(**inputs, generation_config=temp_generation_config_3, tokenizer=processor.tokenizer, max_new_tokens=2048)
        temperature_decodes_3 = processor.batch_decode(temperature_output_3, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)

        temperature_output_4 = model.generate(**inputs, generation_config=temp_generation_config_4, tokenizer=processor.tokenizer, max_new_tokens=2048)
        temperature_decodes_4 = processor.batch_decode(temperature_output_4, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)

        greedy_output = model.generate(**inputs, generation_config=greedy_generation_config, tokenizer=processor.tokenizer, max_new_tokens=2048)
        greedy_decodes = processor.batch_decode(greedy_output, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)


        for j, data in enumerate(batch_queries):
            final_response.append({'text': data['text'],
                                   'image': data['image'],
                                   'image_path': data['image_path'],
                                   'temp_02': temperature_decodes_1[j],
                                   'temp_04': temperature_decodes_2[j],
                                   'temp_06': temperature_decodes_3[j],
                                   'temp_08': temperature_decodes_4[j],
                                   'greedy': greedy_decodes[j]})

    dump_to_jsonl(final_response, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--llava_data_pth", type=str, default="./pretrain_value_image_9k.jsonl")
    parser.add_argument("--image_folder", type=str, default="./train2017")
    parser.add_argument("--output_file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--per_gpu_batch_size", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    main(args)
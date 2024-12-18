# VisVM
Official codebase for paper "Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension".

## Prepare environment
```
conda env create -f environment.yml

cp ./utils/modeling_llava_next.py ~/.conda/envs/visvm/lib/python3.10/site-packages/transformers/models/llava_next/
cp ./utils/trainer/td_trainer.py ~/.conda/envs/visvm/lib/python3.10/site-packages/trl/trainer/
cp ./utils/__init__.py ~/.conda/envs/visvm/lib/python3.10/site-packages/trl/
cp ./utils/trainer/__init__.py ~/.conda/envs/visvm/lib/python3.10/site-packages/trl/trainer/
```

## Generate responses:
```
bash ./script/batch_generate.sh
```
## Prepare TD training data:
```
bash ./script/clip_score.sh
```
## Train value model:
```
bash ./script/clip_score.sh
```
## SFT VLM:
```
bash ./script/train_sft.sh
```
#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=2

model_name_or_path=/home/sunchao/MODEL_CKPT
data_path=data/objhal_bench.jsonl
save_path=objhal_bench_answer.jsonl

python ./eval/model_eval.py \
--model-name-or-path $model_name_or_path \
--question-file $data_path \
--answers-file $save_path \
--sampling

# python ./eval/model_eval.py \
# --model-name-or-path $model_name_or_path \
# --question-file $data_path \
# --answers-file $save_path \

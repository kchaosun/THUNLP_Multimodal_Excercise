MODEL_NAME="/home/sunchao/THUNLP_Multimodal_Excercise"

echo "========================="
echo "MODEL: $MODEL_NAME"
echo "========================="

python eval/chair.py \
--coco_path data/annotations \
--cache data/chair_300.pkl \
--cap_file $MODEL_NAME/objhal_bench_answer.jsonl \
--save_path $MODEL_NAME/eval-chair-300_answer.json \
--caption_key answer

script_dir=$(readlink -f "$0")
current_dir=$(dirname "$script_dir")
model_path=/alidata/models/BAAI/bge-large-zh-v1.5
# model_name=/home/star/models/bge-large-zh-v1.5
echo "current_dir：$current_dir\n model_name：$model_name"


for var in 'merged_data_train' 'supervised_finetune_qa_train' 'supervised_finetune_knowledge_train' 'uniem_qa_train'
do
    python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path $model_path \
    --input_file $current_dir/data/$var.jsonl \
    --output_file $current_dir/data/$var'_minedHN.jsonl' \
    --range_for_sampling 2-200 \
    --use_gpu_for_searching
done

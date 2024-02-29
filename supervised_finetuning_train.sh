script_dir=$(readlink -f "$0")
current_dir=$(dirname "$script_dir")


# ---!!!!!change it accord to your own config !!!!!-----
model_name=bge-large-zh-v1.5
base_model_dir="/alidata/models/BAAI"    # "/alidata/models/BAAI"   # "/home/pinming/models"
output_model_dir="/alidata/models/BAAI"  # "/alidata/models/BAAI" #"home/pinming/models"
finetune_flag="-finetune-v1.0"
num_gpus=2
# -----!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------
model_path=$base_model_dir/$model_name

model_output_path=$output_model_dir/$model_name$finetune_flag

echo "current_dir：$current_dir base_model_path:$model_path model_output_path: $model_output_path"

# RTX 3090 or 4000 series doesn't support faster communication broadband via P2P or IB
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

echo '-----------------------start to process data ...-------------------'

bash data_process.sh

echo '-----------------------start to select hard negative samples...-----'

for var in 'merged_data_train' 'supervised_finetune_qa_train' 'supervised_finetune_knowledge_train' 'uniem_qa_train'

do
    python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path $model_path \
    --input_file $current_dir/data/$var.jsonl \
    --output_file $current_dir/data/$var'_minedHN.jsonl' \
    --range_for_sampling 2-200 \
    --use_gpu_for_searching
done

# all options in training and evaluation
#-----------------train options---------------------------------------------------------------------------------------
data_sources=('merged_data_train' 'supervised_finetune_qa_train' 'supervised_finetune_knowledge_train' 'uniem_qa_train')
instructions=("''" '为这个句子生成表示以用于检索相关文章：')
instruct_flags=("" "--instruction_flag")
hardneg_flags=('' '_minedHN')

data_source=${data_sources[3]}
hardneg=${hardneg_flags[1]}
eval_data='merge_qa_eval'
#---------------train options ----------------------------------------------------------------------------------------

echo '-----------------------start to train and eval...--------------------'
echo '+++++++++++++++++++++++related params:+++++++++++++++++++++++++++++++++'
echo all the params are: instruction:${instruction[0]}, hardneg:$hardneg, data_source:$data_source, instruct_flag:${instruct_flag[0]}
echo '+++++++++++++++++++++++start to train++++++++++++++++++++++++++++++++++'
torchrun \
--nproc_per_node $num_gpus \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]} \
--model_name_or_path $model_path \
--save_steps 1000000 \
--train_data $current_dir/data/$data_source$hardneg.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 5 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 256 \
--passage_max_len 512 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval ${instructions[0]} 
echo '+++++++++++++++++++++++training done+++++++++++++++++++++++++++++++++++++'
echo '+++++++++++++++++++++++start to eval ....++++++++++++++++++++++++++++++++'
echo "-----------------------finetuned model path $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]}-----------"
#----------------------------with rerank----------------------------------------------------------
python eval_llama_index.py --file_name_reg $eval_data \
                            --model_output_path $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]} \
                            ${instruct_flags[0]} \
                            --eval_model_name original >$data_source'_hardneg:'$hardneg'_instruction_flag:'${instruct_flags[0]}'_original_reranker.log'

python eval_llama_index.py --file_name_reg $eval_data \
                            --model_output_path $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]} \
                            ${instruct_flags[0]} \
                            --eval_model_name tianrang >$data_source'_hardneg:'$hardneg'_instruction_flag:'${instruct_flags[0]}'_tianrang_reranker.log'

python eval_llama_index.py --file_name_reg $eval_data \
                            --model_output_path $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]} \
                            ${instruct_flags[0]} \
                            --eval_model_name finetuned >$data_source'_hardneg:'$hardneg'_instruction_flag:'${instruct_flags[0]}'_finetuned_reranker.log';
# -----------------------------------without reranker ----------------------------------------------
python eval_llama_index.py --file_name_reg $eval_data --disable_reranker \
                            --model_output_path $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]} \
                            ${instruct_flags[0]} \
                            --eval_model_name original >$data_source'_hardneg:'$hardneg'_instruction_flag:'${instruct_flags[0]}'_original.log'

python eval_llama_index.py --file_name_reg $eval_data --disable_reranker \
                            --model_output_path $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]} \
                            ${instruct_flags[0]} \
                            --eval_model_name tianrang >$data_source'_hardneg:'$hardneg'_instruction_flag:'${instruct_flags[0]}'_tianrang.log'

python eval_llama_index.py --file_name_reg $eval_data --disable_reranker \
                            --model_output_path $model_output_path-data:$data_source-neg:$hardneg-flag:${instruct_flags[0]} \
                            ${instruct_flags[0]} \
                            --eval_model_name finetuned >$data_source'_hardneg:'$hardneg'_instruction_flag:'${instruct_flags[0]}'_finetuned.log';

echo '+++++++++++++++++++++++eval done +++++++++++++++++++++++++++++++++++++++++'
# huggingface-cli download namespace-Pt/msmarco-corpus --resume-download --repo-type datasets  && \
# huggingface-cli download namespace-Pt/msmarco-corpus --resume-download --repo-type dataset  && \
script_dir=$(readlink -f "$0")
current_dir=$(dirname $script_dir)
base_model_name=/alidata/models/BAAI/bge-large-zh
# base_model_name=/home/star/models/bge-large-zh-v1.5
finetuned_model_name=$current_dir/model_output/supervised_train 
private_eval_path=$current_dir/data/supervised_finetune_data.jsonl
msmarco_path=$current_dir/eval_supervised_msmarco.py

echo "current_dir：$current_dir\nbase_model_name：$base_model_name\nfinetuned_model_name:$finetuned_model_name"

# -------------------------------------------------------
# eval finetuned model on public data 
python $msmarco_path \
--encoder $finetuned_model_name \
--fp16 \
--add_instruction \
--k 100 >finetuned_msmarco.log 2>&1 

# eval finetuned model on private data 
python $msmarco_path \
--encoder $finetuned_model_name \
--fp16 \
--add_instruction \
--private_eval_path $private_eval_path \
--k 100 >finetuned_private.log 2>&1

# eval finetuned model on private data, but add some public data as candidate corpus 
python $msmarco_path \
--encoder $finetuned_model_name \
--fp16 \
--add_instruction \
--private_eval_path $private_eval_path \
--add_extra_corpus \
--k 100 >finetuned_private_extra_corpus.log 2>&1 

# -------------------------------------------------------
# eval base model on public data 
python $msmarco_path \
--encoder $base_model_name \
--fp16 \
--add_instruction \
--k 100 >base_msmarco.log 2>&1 

# eval base model on private data 
python $msmarco_path \
--encoder $base_model_name \
--fp16 \
--add_instruction \
--private_eval_path $private_eval_path \
--k 100 >base_private.log 2>&1

# eval base model on private data, but add some public data as candidate corpus 
python $msmarco_path \
--encoder $base_model_name \
--fp16 \
--add_instruction \
--private_data_path $private_eval_path \
--add_extra_corpus \
--k 100 >base_private_extra_corpus.log 2>&1 

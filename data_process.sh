script_dir=$(readlink -f "$0")
current_dir=$(dirname "$script_dir")
echo current dir: $current_dir
# ---------------knowledge_engineering------------
echo 'start to extract knowledge engineering...'

python batch_knowledge_engineering.py --app_type QA --keep_items 1
# python batch_knowledge_engineering.py --app_type SUMMARY

echo 'start to process uniem qa data...'
# -----------------process uniem data ---------------------------
python data_processor.py --data_dir $current_dir/data/ \
                         --file_name_reg 'datasets_qa_v1.0.jsonl' \
                         --data_source uniem_qa \
                         --output_file_name uniem_qa > uniem_qa.log 2>&1 
# -----------------train data process ---------------------------
echo "start to process training data..."
python data_processor.py --process_func supervised_data_process \
                        --combine_flag \
                        --data_dir $current_dir/data/ \
                        --process_args '{"file_name":"supervised_finetune_knowledge_train.jsonl"}' \
                        --return_amount train  \
                        --data_source knowledge_engineering >train_knowledge.log 2>&1 

python data_processor.py --data_dir $current_dir/data/datasets/ \
                        --file_name_reg '.*' \
                        --output_file_name supervised_finetune_qa_train.jsonl \
                        --return_amount train \
                        --data_source qa_generated >train_qa.log 2>&1

python data_processor.py --data_source merge_all \
                            --file_name_reg "^(?!.*merged).*train.jsonl" \
                            --data_dir $current_dir/data \
                            --output_file_name merged_data_train.jsonl >train_merge_all.log 2>&1

#----------------eval data process--------------------------------------------------------
echo 'start to process eval data...'
python data_processor.py --process_func supervised_data_process \
                                --combine_flag \
                                --data_dir $current_dir/data/ \
                                --process_args '{"file_name":"supervised_finetune_knowledge_eval.jsonl"}' \
                                --return_amount eval  \
                                --data_source knowledge_engineering >eval_knowledge.log 2>&1 

python data_processor.py --data_dir $current_dir/data/datasets \
                        --file_name_reg '.*' \
                        --output_file_name supervised_finetune_qa_eval.jsonl \
                        --return_amount eval \
                        --data_source qa_generated >eval_qa.log 2>&1

python data_processor.py --data_source merge_all \
                            --file_name_reg "^(?!.*merged).*eval.jsonl" \
                            --data_dir $current_dir/data/ \
                            --output_file_name merged_data_eval.jsonl >eval_merge_all.log 2>&1

import __init__
from enum import Enum
from evaluate.evaluate_db_vector import EvaluateDBVector
from evaluate.evaluate_llama_index import EvaluateLlamaIndex
from evaluate.evaluate_llama_index_new import EvaluateLlamaIndexNew


class EvaluateMethodType(str, Enum):
    db_vector = 'db_vector'
    llama_index = 'llama_index'


class EvaluateModel():
    def evaluate_m3e(self, train_params, finetune_model_path, json_file_name="", evaluate_method_type: EvaluateMethodType = EvaluateMethodType.db_vector, evaluate_base=True):
        if evaluate_method_type == EvaluateMethodType.db_vector:
            EvaluateDBVector.evaluate_m3e(train_params, finetune_model_path)
        elif evaluate_method_type == EvaluateMethodType.llama_index:
            if json_file_name.endswith('.jsonl'):
                EvaluateLlamaIndex.evaluate_m3e(train_params, finetune_model_path, json_file_name)
            elif json_file_name.endswith('.json'):
                EvaluateLlamaIndexNew.evaluate_m3e(train_params, finetune_model_path, json_file_name, evaluate_base)
            else:
                raise Exception(f"The file has an unrecognized extension in evaluate_m3e,json_file_name:{json_file_name}")

    def evaluate_bge(self, train_params, finetune_model_path, json_file_name="", evaluate_method_type: EvaluateMethodType = EvaluateMethodType.db_vector, evaluate_base=True):
        if evaluate_method_type == EvaluateMethodType.db_vector:
            EvaluateDBVector.evaluate_bge(train_params, finetune_model_path)
        elif evaluate_method_type == EvaluateMethodType.llama_index:
            if json_file_name.endswith('.jsonl'):
                EvaluateLlamaIndex.evaluate_bge(train_params, finetune_model_path, json_file_name)
            elif json_file_name.endswith('.json'):
                EvaluateLlamaIndexNew.evaluate_bge(train_params, finetune_model_path, json_file_name, evaluate_base)
            else:
                raise Exception(f"The file has an unrecognized extension in evaluate_bge,json_file_name:{json_file_name}")

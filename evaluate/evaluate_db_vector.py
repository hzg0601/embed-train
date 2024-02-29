import __init__

from evaluate.db_operate_specification import update_pms_m3e_vector, update_pms_bge_large_vector
from evaluate.test_parse_excel import make_pms_m3e_vector_content, make_pms_bge_large_vector_content


class EvaluateDBVector():
    @classmethod
    def evaluate_m3e(self, train_params, finetune_model_path):
        # 更新数据库中的向量
        update_pms_m3e_vector(finetune_model_path)

        # 生成excel中的数据
        make_pms_m3e_vector_content(train_params, finetune_model_path)
    
    @classmethod
    def evaluate_bge(self, train_params, finetune_model_path):
        # 更新数据库中的向量
        update_pms_bge_large_vector(finetune_model_path)

        # 生成excel中的数据
        make_pms_bge_large_vector_content(train_params, finetune_model_path)

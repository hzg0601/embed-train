import json
import os

import __init__

from evaluate.db_connect_pg import DBConnectPG

from sentence_transformers import SentenceTransformer


# 数据库操作父类
class DBOperateBase(DBConnectPG):

    m3ebase_model = None
    pms_m3ebase_model = None

    bgelarge_model = None
    pms_bgelarge_model = None

    # 微调模型列表，根据路径判断，控制更新向量等时，只加载一次
    finetune_models = {}

    def to_vector_m3ebase(self, text):
        if self.m3ebase_model is None:
            self.m3ebase_model = SentenceTransformer(self.m3e_embedding_name)

            float_array = self.m3ebase_model.encode(text)
        return json.dumps(float_array.tolist())

    def to_vector_pms_m3ebase(self, text, finetune_model_path=""):
        if finetune_model_path:
            if finetune_model_path not in self.finetune_models:
                self.finetune_models[finetune_model_path] = SentenceTransformer(finetune_model_path)
            pms_m3ebase_model = self.finetune_models[finetune_model_path]
        else:
            if self.pms_m3ebase_model is None:
                self.pms_m3ebase_model = SentenceTransformer(self.pms_m3e_embedding_name)
            pms_m3ebase_model = self.pms_m3ebase_model

        float_array = pms_m3ebase_model.encode(text)
        return json.dumps(float_array.tolist())

    def to_vector_bgelarge(self, text, question=False):
        from FlagEmbedding import FlagModel
        if self.bgelarge_model is None:
            self.bgelarge_model = FlagModel(self.bgelarge_embedding_name, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")

        # 参考使用说明，数据库中提前生成的向量，和提问转向量有所不同，https://huggingface.co/BAAI/bge-base-zh
        if question:
            float_array = self.bgelarge_model.encode_queries(text)
        else:
            float_array = self.bgelarge_model.encode(text)
        return json.dumps(float_array.tolist())

    def to_vector_pms_bgelarge(self, text, question=False, finetune_model_path=""):
        from FlagEmbedding import FlagModel

        if finetune_model_path:
            if finetune_model_path not in self.finetune_models:
                self.finetune_models[finetune_model_path] = FlagModel(
                    self.pms_bgelarge_embedding_name, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
            pms_bgelarge_model = self.finetune_models[finetune_model_path]
        else:
            if self.pms_bgelarge_model is None:
                self.pms_bgelarge_model = FlagModel(self.pms_bgelarge_embedding_name, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
            pms_bgelarge_model = self.pms_bgelarge_model

        # 参考使用说明，数据库中提前生成的向量，和提问转向量有所不同，https://huggingface.co/BAAI/bge-base-zh
        if question:
            float_array = pms_bgelarge_model.encode_queries(text)
        else:
            float_array = pms_bgelarge_model.encode(text)
        return json.dumps(float_array.tolist())

    def to_vector(self, text, vector_field_name, question=False, finetune_model_path=""):
        if vector_field_name.startswith('m3e_'):
            return self.to_vector_m3ebase(text)
        if vector_field_name.startswith('pms_m3e_'):
            return self.to_vector_pms_m3ebase(text, finetune_model_path)
        elif vector_field_name.startswith('bge_large_'):
            return self.to_vector_bgelarge(text, question)
        elif vector_field_name.startswith('pms_bge_large_'):
            return self.to_vector_pms_bgelarge(text, question, finetune_model_path)
        else:
            raise Exception(
                f"vector_field_name:{vector_field_name} is not supported")

    def __init__(self, config_prefix):  # sourcery skip: raise-specific-error
        super().__init__(config_prefix)

        self.m3e_embedding_name = os.environ.get("M3E_EMBEDDINGS_MODEL_PATH")
        self.pms_m3e_embedding_name = os.environ.get("M3E_FINETUNE_MODEL_PATH")

        self.bgelarge_embedding_name = os.environ.get("BGE_EMBEDDINGS_MODEL_PATH")
        self.pms_bgelarge_embedding_name = os.environ.get("BGE_FINETUNE_MODEL_PATH")


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    pass

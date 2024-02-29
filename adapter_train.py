""" 用于基于非对称标注数据的adapter微调，基于llama-index实现"""
import os
from tools.utils import logger
from typing import Union
from llama_index.finetuning import (SentenceTransformersFinetuneEngine,
                                    EmbeddingAdapterFinetuneEngine)
from llama_index.embeddings import (resolve_embed_model,
                                    AdapterEmbeddingModel,
                                    HuggingFaceEmbedding,)

from llama_index.embeddings.adapter_utils import TwoLayerNN

from data_processor import jsonl2li_data_process
from eval_llama_index import li_eval_finetune
import argparse
from tools.utils import GLOBAL

os.environ["TRANSFORMERS_OFFLINE"] = '1'
EMBED_MODEL_PATH = f"local:{GLOBAL.BGE_BASE_MODEL_PATH}"
model_output_path = f"local:{GLOBAL.BGE_FINETUNE_MODEL_PATH}"


parser = argparse.ArgumentParser(description="The evaluation script that use llama index")

parser.add_argument("--finetune_class",type=str,default="full",choices=['full','adapter'],
                    help="the way to train embedding model, adapter or full")
parser.add_argument("--adapter_class",type=str,default="TwoLayerNN",choices=["Linear","TwoLayerNN"],
                    help="the adapter used in adapter finetune")
parser.add_argument("--model_name",type=str,default=EMBED_MODEL_PATH,
                    help="the model name or path of embedding model")
parser.add_argument("--model_output_path",default=GLOBAL.BGE_FINETUNE_MODEL_PATH,type=str,
                    help="the checkpoint to finetuned model")
parser.add_argument("--data_dir",default=GLOBAL.DATA_PATH,type=str,
                    help="the data path")
parser.add_argument("--file_name_reg",type=str,default="merged_data_train.jsonl",
                    help="the regex expression of data file name")

def li_train_finetune(model_name:str=EMBED_MODEL_PATH,
                  model_output_path:str=GLOBAL.BGE_FINETUNE_MODEL_PATH, # 保存最终的checkpoint的路径
                  model_checkpoint_path:str=None, # 保存中间checkpoint文件的路径
                  engine_class:str="adapter", # adapter,sentence_transformers
                  adapter_model:str=None, # TwoLayerNN, Linear
                  file_name_reg:str="merged_data_train.jsonl",
                  data_dir:str=GLOBAL.DATA_PATH
                  ):
    logger.info("llama-index adapter fine-tuning start...")

    data = jsonl2li_data_process(data_dir=data_dir,file_name_reg=file_name_reg)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger.info("data procession done..")
    # resolve_embed_model(model_name),model_name必须是local: repo_id的格式
    base_embed_model = resolve_embed_model(model_name)
    if adapter_model is not None:
        adapter_model = eval(adapter_model)
        in_features_dim = base_embed_model._model.config.hidden_size
        adapter_model_ins = adapter_model(
        in_features=in_features_dim,  # input dimension
        hidden_features=in_features_dim*3,  # hidden dimension
        out_features=in_features_dim,  # output dimension
        bias=True,
        add_residual=True,
            )
    else:
        adapter_model_ins = None    

    if engine_class == "adapter":
        finetune_engine = EmbeddingAdapterFinetuneEngine(
            data,
            base_embed_model,
            model_output_path=model_output_path,
            model_checkpoint_path=model_checkpoint_path,
            adapter_model=adapter_model_ins,
            epochs=25,
            verbose=True,
        )
    else:
        finetune_engine = SentenceTransformersFinetuneEngine(
            data,
            model_id=model_name,
            model_output_path=model_output_path
        )
    logger.info("fine-tuning start...")
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model(
    adapter_cls=adapter_model
    )
    logger.info("llama-index adapter fine-tuning done.")
    return embed_model

def train_eval_pipeline(finetune_class:str="adapter", # adapter,sentence_transformers
                  adapter_class: str="Linear", # None for linear, "TwoLayerNN"
                  model_name:str=EMBED_MODEL_PATH,
                  model_output_path:str=GLOBAL.BGE_FINETUNE_MODEL_PATH,
                  use_instruction:bool=False):
    
    li_train_finetune(
                     model_name=model_name,
                     model_output_path=model_output_path,
                     engine_class=finetune_class,
                     adapter_model=adapter_class
                     )
    
    li_eval_finetune(
                     finetune_class=finetune_class,
                     adapter_class=adapter_class,
                     model_name=model_name,
                     model_output_path=model_output_path
                     )
    
if __name__ == "__main__":
    li_train_finetune()
    li_eval_finetune()
    

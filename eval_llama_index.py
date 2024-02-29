import os
import pandas as pd
import asyncio
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms import HuggingFaceLLM
from llama_index.schema import TextNode
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import (resolve_embed_model,
                                    AdapterEmbeddingModel,
                                    HuggingFaceEmbedding,)
from llama_index.evaluation import RetrieverEvaluator
from data_processor import jsonl2li_data_process
from tianrang_embedding import TianrangLIEmbedding
import torch
from llama_index.postprocessor import SentenceTransformerRerank
from tools.utils import GLOBAL
import warnings
from tools.logger import getLogger

logger = getLogger()

warnings.filterwarnings("ignore")
EMBED_MODEL_PATH = GLOBAL.BGE_BASE_MODEL_PATH
FINETUNE_MODEL_PATH = GLOBAL.BGE_FINETUNE_MODEL_PATH  # 注意与训练脚本supervised_finetuning_train, batch_train_eval.sh的路径名一致
LLM_MODEL_PATH = GLOBAL.EVALUATE_MODEL_PATH
RERANKER_PATH = GLOBAL.RERANKER_MODEL_PATH

parser = argparse.ArgumentParser(description="The evaluation script that use llama index")

parser.add_argument("--finetune_class", type=str, default="full", choices=['full', 'adapter'],
                    help="the way to train embedding model, adapter or full")
parser.add_argument("--adapter_class", type=str, default="TwoLayerNN", choices=["Linear", "TwoLayerNN"],
                    help="the adapter used in adapter finetune")
parser.add_argument("--model_name", type=str, default=EMBED_MODEL_PATH,
                    help="the model name or path of embedding model")
parser.add_argument("--model_output_path", default=FINETUNE_MODEL_PATH, type=str,
                    help="the checkpoint to finetuned model")
parser.add_argument("--data_dir", default=GLOBAL.DATA_PATH, type=str,
                    help="the data path")
parser.add_argument("--file_name_reg", type=str, default="merged_data_eval.jsonl",
                    help="the regex expression of data file name")
parser.add_argument("--instruction_flag", action="store_true", default=False,
                    help="add instruction in evaluation")
parser.add_argument("--eval_model_name", type=str, choices=["tianrang", "original", "finetuned", "all"], default="all",
                    help="the model to eval, `all` for all models")
parser.add_argument("--disable_reranker", action="store_true", default=False,
                    help="use reraker model to rerank docs")
parser.add_argument("--reranker_model_name", type=str, default=RERANKER_PATH,
                    help="the path to reranker model")


def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
    full_df = pd.DataFrame(metric_dicts)
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    metric_df = pd.DataFrame(
        {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )
    return metric_df
    # return metric_df


def eval_embedding(embed_model,
                   llm,
                   nodes,
                   data,
                   reranker_model_name=None,
                   flag="finetuned",
                   eval_ks=[1, 2, 3, 5, 10]):
    """
    Evaluate the embeddings of nodes by retrieving similar nodes.

    Args:
        embed_model (str): The name of the embedding model.
        llm (object): The language model object.
        nodes (list): The list of nodes to evaluate.
        data (list): The list of datasets to evaluate.
        reranker_model_name (str, optional): The name of the reranking model. Defaults to None.
        flag (str, optional): The flag of the embedding model. Defaults to "finetuned".
        eval_ks (list, optional): The list of k values to evaluate. Defaults to [1, 2, 3, 5, 10].

    Returns:
        None
    """

    logger.info(f"----------------eval_embedding start {flag}----------------")
    torch.cuda.empty_cache()
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    vector_index = VectorStoreIndex(nodes, service_context=service_context)
    # vector_index在执行操作时会产生cuda的cache
    # todo 确认cuda cache的来源
    torch.cuda.empty_cache()
    for k in eval_ks:
        logger.info(f"----------------eval_embedding k={k} {flag}----------------")
        if reranker_model_name is not None:
            reranker = SentenceTransformerRerank(top_n=k, model=reranker_model_name)
        else:
            reranker = None
        retriever = vector_index.as_retriever(similarity_top_k=k, postprocessor=reranker)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )
        eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(data))
        metric_df = display_results(f"top-{k} eval", eval_results)
        torch.cuda.empty_cache()
        print(f"----------------performance of {flag}----------------")
        print(metric_df)
        logger.warning(f"----------------performance of {flag}----------------")
        logger.warning(metric_df)


def li_eval_finetune(finetune_class: str = "adapter",  # adapter,full
                     adapter_class: str = "Linear",  # None for linear, "TwoLayerNN"
                     model_name: str = EMBED_MODEL_PATH,
                     model_output_path: str = FINETUNE_MODEL_PATH,
                     file_name_reg: str = "merged_data_eval.jsonl",
                     data_dir: str = GLOBAL.DATA_PATH,
                     instruction_flag: bool = False,
                     disable_reranker: bool = False,
                     reranker_model_name: str = RERANKER_PATH,
                     train_params: str = "",
                     eval_ks=[1, 2, 3, 5, 10]
                     ):
    """ 以异步的方式按照llama-index的retriever模式评估模型的表现
    para@finetune_class: 
    """
    data = jsonl2li_data_process(data_dir=data_dir,
                                 file_name_reg=file_name_reg,
                                 instruction_flag=instruction_flag)
    nodes = [TextNode(text=value, id_=key) for key, value in data.corpus.items()]

    # 针对qwen-14-int4,注意要在config.json中的quantization_config中加入disable_exllama:true
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
    del model
    torch.cuda.empty_cache()

    if disable_reranker:
        reranker_model_name = None

    # 天壤模型方式评估，速度慢，先关闭
    # tianrang_embed = TianrangLIEmbedding()
    # print("start to eval tianrang model ...")
    # eval_embedding(tianrang_embed,llm,nodes,data,reranker_model_name,flag="tianrang")

    if finetune_class == "adapter":
        base_embed_model = resolve_embed_model(model_name)

        print("start to eval base model ...")
        eval_embedding(base_embed_model, llm, nodes, data, reranker_model_name, flag="original", eval_ks=eval_ks)
        del base_embed_model
        torch.cuda.empty_cache()

        adapter_class = eval(adapter_class) if adapter_class == "TwoLayerNN" else None
        embed_model = AdapterEmbeddingModel(
            base_embed_model=base_embed_model,
            adapter_path=model_output_path,
            adapter_cls=adapter_class
        )

        logger.info(f"start to eval finetuned model(train_params:{train_params}) ....")
        eval_embedding(embed_model, llm, nodes, data, reranker_model_name, flag="finetuned", eval_ks=eval_ks)
    else:
        base_embed_model = HuggingFaceEmbedding(model_name=model_name)
        logger.info("start to eval base model ...")
        eval_embedding(base_embed_model, llm, nodes, data, reranker_model_name, flag="original", eval_ks=eval_ks)
        del base_embed_model
        torch.cuda.empty_cache()

        embed_model = HuggingFaceEmbedding(model_name=model_output_path)
        logger.info(f"start to eval finetuned model(train_params:{train_params}) ....")
        eval_embedding(embed_model, llm, nodes, data, reranker_model_name, flag="finetuned", eval_ks=eval_ks)

    logger.info("all eval done")


def eval_single_model(
    finetune_class: str = "adapter",  # adapter,full
    adapter_class: str = "Linear",  # None for linear, "TwoLayerNN"
    model_name: str = EMBED_MODEL_PATH,
    model_output_path: str = FINETUNE_MODEL_PATH,
    file_name_reg: str = "merged_data_eval.jsonl",
    data_dir: str = GLOBAL.DATA_PATH,
    instruction_flag: bool = False,
    eval_model_name: str = "tianrang",
    disable_reranker: bool = False,
    reranker_model_name: str = RERANKER_PATH
):
    data = jsonl2li_data_process(data_dir=data_dir,
                                 file_name_reg=file_name_reg,
                                 instruction_flag=instruction_flag)
    nodes = [TextNode(text=value, id_=key) for key, value in data.corpus.items()]

    # 针对qwen-14-int4,注意要在config.json中的quantization_config中加入disable_exllama:true
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
    del model
    if disable_reranker:
        reranker_model_name = None

    torch.cuda.empty_cache()
    if eval_model_name == "tianrang":
        embed_model = TianrangLIEmbedding()
    elif eval_model_name == "original":
        if finetune_class == "adapter":
            embed_model = resolve_embed_model(model_name)
        else:
            embed_model = HuggingFaceEmbedding(model_name=model_name)
    elif eval_model_name == "finetuned":
        if finetune_class == "adapter":
            adapter_class = eval(adapter_class) if adapter_class == "TwoLayerNN" else None
            base_embed_model = resolve_embed_model(model_name)
            embed_model = AdapterEmbeddingModel(
                base_embed_model=base_embed_model,
                adapter_path=model_output_path,
                adapter_cls=adapter_class
            )
        else:
            embed_model = HuggingFaceEmbedding(model_name=model_output_path)

    logger.info(f"start to eval {eval_model_name} model ....")
    eval_embedding(embed_model, llm, nodes, data, reranker_model_name, flag=eval_model_name)
    logger.info(f"eval {eval_model_name} done.")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.eval_model_name == "all":
        li_eval_finetune(
            finetune_class=args.finetune_class,  # adapter,full
            adapter_class=args.adapter_class,  # None for linear, "TwoLayerNN"
            model_name=args.model_name,
            model_output_path=args.model_output_path,
            file_name_reg=args.file_name_reg,
            data_dir=args.data_dir,
            instruction_flag=args.instruction_flag
        )
    else:
        eval_single_model(
            finetune_class=args.finetune_class,  # adapter,full
            adapter_class=args.adapter_class,  # None for linear, "TwoLayerNN"
            model_name=args.model_name,
            model_output_path=args.model_output_path,
            file_name_reg=args.file_name_reg,
            data_dir=args.data_dir,
            instruction_flag=args.instruction_flag,
            eval_model_name=args.eval_model_name
        )

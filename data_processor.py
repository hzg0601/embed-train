""" 用于数据处理 """
import os
import re
import uuid
import pandas as pd
import numpy as np
from llama_index.finetuning import EmbeddingQAFinetuneDataset
import json
import argparse
from tools.utils import logger
from tqdm import tqdm

from tools.utils import GLOBAL

parser = argparse.ArgumentParser(description="data preprocessor")

parser.add_argument("--process_func", default="supervised_data_process", type=str,
                    choices=["adapter_data_process", "unsupervised_data_process", "supervised_data_process"],
                    help="the unit function to preprocess data")
parser.add_argument("--combine_flag", action="store_true", default=True,
                    help="combine all data or not")
parser.add_argument("--instruction_flag", action="store_true", default=False,
                    help="use instruction or not,default False")
parser.add_argument("--process_args", type=str, default='{"file_name":"supervised_finetune_data_eval.jsonl"}',
                    help="the args of process_func,pass it as a str")

parser.add_argument("--return_amount", type=str, default="train", choices=['train', 'full', 'eval'],
                    help="the data collection to return,one of 'train','full','eval'")

parser.add_argument("--data_source", default="uniem_qa", type=str,
                    choices=["qa_generated", "knowledge_engineering", "merge_all","uniem_qa"],
                    help="if merge_all, then merge all available jsonl that satisfies file_name_reg,else load data source")

parser.add_argument("--output_dir", type=str, default=GLOBAL.DATA_PATH)
parser.add_argument("--output_file_name", type=str, default="supervised_finetune_qa_train.jsonl")

parser.add_argument("--data_dir", type=str, default=GLOBAL.DATA_PATH+"/datasets/", help="the source data dir")
parser.add_argument("--file_name_reg", type=str, default="*", help="the regex expression of file name")


np.random.seed(123)


def data_reader(
    path=GLOBAL.DATA_PATH+"/knowledge_qa/",
    file_keyword='QA',
    id_cols=["文档id", "切片id"],
    content_cols=["切片内容", "Q"]
):
    """读取file_keyword指定文件的数据，并取content_cols指定的列
    """
    data_files = [os.path.join(path, file)
                  for file in os.listdir(path)
                  if re.search(file_keyword, file)]
    data_list = []
    for file in data_files:
        data = pd.read_excel(file)[id_cols+content_cols].dropna(subset=["切片内容"])
        data_list.append(data)

    data_list = pd.concat(data_list)[content_cols]

    return data_list


def data_dict_gen(
    path_dir_list=(GLOBAL.DATA_PATH+"/knowledge_qa/",)*2,
    file_keyword_list=("QA", "SUMMARY"),
    id_cols_list=(["文档id"],)*2,
    content_cols_list=(["切片内容", "Q", "切片id"],
                       #    ["切片内容","A","切片id"],
                       ["切片内容", "摘要", "切片id"]),
    cols_map={"切片id": "doc_id", "Q": "query",
              "切片内容": "doc", "A": "query",
              "摘要": "query"
              }
):
    """用于读取并做基本处理"""
    data_dict = {}
    key_list = ["content_query", "content_summary"]  # "content_answer",
    for key, path, file_keyword, id_cols, content_cols in zip(
        key_list,
        path_dir_list,
        file_keyword_list,
        id_cols_list,
        content_cols_list
    ):
        try:
            data_dict[key] = data_reader(path=path,
                                         file_keyword=file_keyword,
                                         id_cols=id_cols,
                                         content_cols=content_cols).rename(columns=cols_map)
        except Exception as e:
            logger.warning(f"key {key}, file keyword {file_keyword}, id_cols {id_cols}, content_cols,{content_cols}")
            logger.warning(e)
    return data_dict


def add_instruction(data: pd.DataFrame, instruction: str = None):
    if instruction:
        data['query'] = instruction + ":" + data['query']
    return data


def adapter_data_process(data: pd.DataFrame):
    """用于生成llama-index的adapter tuning使用的数据格式
    nodes_dict, queries_dict, relevant_docs
    确保列名为：doc,query,doc_id
    """

    nodes_dict, queries_dict, relevant_docs = {}, {}, {}
    for idx, row in data.iterrows():
        question_id = str(uuid.uuid4())
        doc_id = row["doc_id"]
        doc = row["doc"]
        query = row["query"]
        nodes_dict[doc_id] = doc
        queries_dict[question_id] = query
        relevant_docs[question_id] = [doc_id]  # 此处要求doc_id为一个list
    result = EmbeddingQAFinetuneDataset(
        queries=queries_dict,
        corpus=nodes_dict,
        relevant_docs=relevant_docs
    )
    # 按queries:queries_dict, corpus:node_dict, relevant_docs:relevant_docs的方式存为json
    # result.save_json("test.json",encode="utf-8",indent=4)
    return result


def jsonl2li_data_process(data_dir=GLOBAL.DATA_PATH,
                          file_name_reg="merged_data_eval.jsonl",
                          instruction_flag=False):
    """将 {query, pos, neg}的jsonl数据转换为llama-index的EmbeddingQAFinetuneDataset"""
    logger.info("start to transform data...")
    file_list = [os.path.join(data_dir, file) for file in
                 os.listdir(data_dir) if re.search(file_name_reg, file)]
    nodes_dict, queries_dict, relevant_docs = {}, {}, {}
    for file in file_list:
        with open(file, "r") as f:
            for line in f:
                line = json.loads(line)
                if not line["pos"]:
                    continue

                question_id = str(uuid.uuid4())
                doc_id = str(uuid.uuid4())
                doc = line["pos"][0]
                query = line["query"] if not instruction_flag else GLOBAL.INSTRUCTION_DICT["default"] + line["query"]
                nodes_dict[doc_id] = doc
                queries_dict[question_id] = query
                relevant_docs[question_id] = [doc_id]  # 此处要求doc_id为一个list
        result = EmbeddingQAFinetuneDataset(
            queries=queries_dict,
            corpus=nodes_dict,
            relevant_docs=relevant_docs
        )
    # 按queries:queries_dict, corpus:node_dict, relevant_docs:relevant_docs的方式存为json
    # result.save_json("test.json",encode="utf-8",indent=4)
    return result


def unsupervised_data_process(data: pd.DataFrame):
    pass


def listjson2trainjson(data_dir=GLOBAL.DATA_PATH+"/datasets/", file_name_reg="*"):
    """ 
    data_dir: str, 存放以uniem格式保存的数据，修改为flagembedding格式的数据
    """
    file_list = [os.path.join(data_dir, file) for file in
                 os.listdir(data_dir) if re.search(file_name_reg, file)]
    data_list = []
    for file in file_list:
        with open(file, 'r') as f:
            data = json.load(f)
        queries = data["queries"]
        corpus = data["corpus"]
        maps = data["relevant_docs"]
        que_id, que, doc_id, doc = [], [], [], []
        for uid, query in queries.items():
            que_id.append(uid)
            que.append(query)
            doc_id.append(maps[uid][0])
            doc.append(corpus[maps[uid][0]])

        data = pd.DataFrame({"query_id": que_id, "query": que,
                             "doc_id": doc_id, "doc": doc})
        data_list.append(data)
    data_list = pd.concat(data_list, axis=0, ignore_index=True)
    return data_list


def supervised_data_process(data: pd.DataFrame,
                            output_dir=GLOBAL.DATA_PATH,
                            file_name="supervised_finetune_data_train.jsonl",
                            n_negs=3):
    logger.info("process unit data")
    result = []
    temp = {}
    output_path = os.path.join(output_dir, file_name)
    file = open(output_path, mode="w", encoding="utf-8")
    total = data.shape[0]
    for i, row in tqdm(data.iterrows(), total=total):
        temp["query"] = row["query"]
        temp["pos"] = [row["doc"]]
        candidate = data[data["doc_id"] != row["doc_id"]]["doc"]
        temp["neg"] = np.random.choice(candidate, n_negs).tolist()
        json.dump(temp, file, ensure_ascii=False)
        file.write("\n")
        result.append(temp)
    file.close()
    logger.info("process unit done.")
    return result


def shuffle_data(data: pd.DataFrame, return_amount: str = "full", ratio: float = 0.75):
    data = data.take(np.random.permutation(data.shape[0]))
    length = int(data.shape[0]*ratio)

    if return_amount == "full":
        return data
    elif return_amount == "train":
        return data[:length]
    else:
        return data[length:]


def union_data_process(process_func=supervised_data_process,
                       combine_flag: bool = True,
                       instruction_flag: str = False,
                       process_args: dict = {"file_name": "supervised_finetune_data_train.jsonl"},
                       return_amount: str = "eval"):
    """用于不同格式数据的统一处理
       process_func: 处理数据格式的函数
       combine_flag: 是否合并返回
       instruction_flag:是否在query中使用instruction
       process_args: process_func的关键字参数
       return_amount:返回数量的量, full, train, eval,以3/4为训练集，1/4为测试集
    """
    logger.info("start to process knowledge engineering data ...")
    data_df_dict = data_dict_gen()
    if instruction_flag:
        for key, data_df in data_df_dict.items():
            data_df_dict[key] = add_instruction(data=data_df, instruction=GLOBAL.INSTRUCTION_DICT[key])

    if combine_flag:
        data_df = pd.concat(list(data_df_dict.values()))
        data_df = shuffle_data(data_df, return_amount=return_amount)
        result = process_func(data_df, **process_args)

    else:
        result = {}
        for key, value in data_df_dict.items():
            value = shuffle_data(value, return_amount=return_amount)
            temp = process_func(value, **process_args)
            result[key] = temp
    logger.info("process knowledge engineering data done.")
    return result


def qa_generated_process(data_dir=GLOBAL.DATA_PATH+"/datasets/",
                         file_name_reg="*",
                         instruct_flag=False,
                         output_dir=GLOBAL.DATA_PATH,
                         output_file_name="supervised_finetune_qa_train.jsonl",
                         return_amount="train",
                         n_negs=3):
    logger.info("start to process qa generated data...")
    data_df = listjson2trainjson(data_dir=data_dir, file_name_reg=file_name_reg)
    if instruct_flag:
        data_df["query"] = GLOBAL.INSTRUCTION_DICT["content_query"]+":"+data_df['query']
    data_df = shuffle_data(data_df, return_amount=return_amount)
    data = supervised_data_process(data=data_df,
                                   output_dir=output_dir,
                                   file_name=output_file_name,
                                   n_negs=n_negs)
    logger.info("process qa generated data done.")
    return data


def uniem_qa_process(data_dir=GLOBAL.DATA_PATH,
                   file_name_reg="datasets_qa_v1.0.jsonl",
                   output_file_name="uniem_qa"):
    logger.info("start to process uniem data...")

    file_list = [os.path.join(data_dir, file) for file in
                 os.listdir(data_dir) if re.search(file_name_reg, file)]
    
    output_path = data_dir + "/" + output_file_name
    target_train = open(output_path+"_train.jsonl", mode="w", encoding="utf-8")
    target_eval = open(output_path+"_eval.jsonl", mode="w", encoding="utf-8")

    for file in file_list:
        with open(file, "r") as f:
            for line in f:
                line = json.loads(line)
                if np.random.random() <= 0.75:
                    json.dump(line, target_train, ensure_ascii=False)
                    target_train.write("\n")
                else:
                    json.dump(line, target_eval, ensure_ascii=False)
                    target_eval.write("\n")                    
    target_train.close()
    target_eval.close()
    logger.info("process unime data done.")


def merge_all_json(data_dir=GLOBAL.DATA_PATH,
                   file_name_reg="^(?!.*merged).*train.jsonl",
                   output_file_name="merged_data_train.jsonl"):
    logger.info("start to merge data...")
    file_list = [os.path.join(data_dir, file) for file in
                 os.listdir(data_dir) if re.search(file_name_reg, file)]
    output_path = data_dir + "/" + output_file_name
    target = open(output_path, mode="w", encoding="utf-8")
    for file in file_list:
        with open(file, "r") as f:
            for line in f:
                line = json.loads(line)
                json.dump(line, target, ensure_ascii=False)
                target.write("\n")
    target.close()
    logger.info("merge data done.")


if __name__ == "__main__":

    args = parser.parse_args()
    if args.data_source == "knowledge_engineering":
        result = union_data_process(
            process_func=eval(args.process_func),
            combine_flag=args.combine_flag,
            instruction_flag=args.instruction_flag,
            process_args=eval(args.process_args),
            return_amount=args.return_amount
        )
    elif args.data_source == "qa_generated":
        result = qa_generated_process(
            data_dir=args.data_dir,
            file_name_reg=args.file_name_reg,
            instruct_flag=args.instruction_flag,
            output_dir=args.output_dir,
            output_file_name=args.output_file_name,
            return_amount=args.return_amount
        )
    elif args.data_source == "uniem_qa":
        uniem_qa_process(data_dir=args.data_dir,
                         file_name_reg=args.file_name_reg,
                         output_file_name=args.output_file_name)
        
    elif args.data_source == "merge_all":
        merge_all_json(data_dir=args.data_dir,
                       file_name_reg=args.file_name_reg,
                       output_file_name=args.output_file_name)
    # jsonl2li_data_process()

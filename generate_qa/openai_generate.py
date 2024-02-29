import json
import os

import httpx
import __init__
from tools.utils import GLOBAL

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.llms import OpenAI
from tools.json_function import json_merge, json_convert_readable
from tools.file_operate import FileOperate


DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
    
以下是背景信息。

---------------------
{context_str}
---------------------

给定上述背景信息和没有先验知识。
基于下面的查询只生成问题。

你是一位老师/教授。你的任务是为即将到来的\
{num_questions_per_chunk}个问题设置一个小测验/考试。
问题应该在文档中性质多样化。
限制问题仅涉及所提供的背景信息。"
"""

num_questions_per_chunk = 2

open_ai_http_proxy = os.environ.get("open_ai_http_proxy")
open_ai_api_key = os.environ.get("open_ai_api_key")


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


# initialize llm
llm = OpenAI(model="gpt-3.5-turbo-1106",
             temperature=0.7,
             max_tokens=2048,
             max_retries=5,
             timeout=600,
             api_key=open_ai_api_key,
             api_base="https://api.openai.com/v1",
             http_client=httpx.Client(
                 proxies=open_ai_http_proxy,
                 transport=httpx.HTTPTransport(local_address="0.0.0.0"),
             ),
             )


def generate_datasets(train_files):
    for train_file in train_files:
        filename = os.path.basename(train_file)
        name_without_extension = os.path.splitext(filename)[0]
        saved_file_name = os.path.join(GLOBAL.DATA_PATH, "datasets", name_without_extension + ".json")

        if not os.path.exists(saved_file_name):

            train_nodes = load_corpus([train_file], verbose=True)
            train_dataset = generate_qa_embedding_pairs(
                nodes=train_nodes, llm=llm, qa_generate_prompt_tmpl=DEFAULT_QA_GENERATE_PROMPT_TMPL, num_questions_per_chunk=num_questions_per_chunk)
            train_dataset.save_json(saved_file_name)

            json_convert_readable(saved_file_name, saved_file_name)


def merge_all_datasets(folder_path, saved_file_name):
    json_files = FileOperate.find_files(folder_path, ".json")
    json_merge(json_files, saved_file_name)
    return os.path.exists(saved_file_name)


def test_merge_all_datasets():
    folder_path = os.path.join(GLOBAL.DATA_PATH, "datasets")
    saved_file_name = os.path.join(GLOBAL.DATA_PATH, "all_datasets.json")
    merge_all_datasets(folder_path, saved_file_name)


def test_generate_datasets():
    train_files = [f"{GLOBAL.DOCS_PATH}/JGJ80-2016施工高处作业安全技术规范.pdf",
                   f"{GLOBAL.DOCS_PATH}/GB 50202-2018 建筑地基工程施工质量验收标准.pdf",
                   f"{GLOBAL.DOCS_PATH}/JGJ120-2012 建筑基坑支护技术规程.docx",
                   ]
    generate_datasets(train_files)


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    test_generate_datasets()
    test_merge_all_datasets()
    pass

import os
import logging
import sys
from dotenv import load_dotenv

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # set logger level

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# fh = logging.FileHandler("run.log",mode='a',encoding='utf-8')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

current_dir = os.path.dirname(os.path.abspath(__file__))

# 读配置文件
config_name = "config.env"
config_path = os.path.join(current_dir, "..", "config", config_name)
if not os.path.exists(config_path):
    raise Exception(f"未找到配置文件，请参照配置模板文件，复制并修改，配置文件名为：{config_name}")

load_dotenv(config_path)


class GLOBAL:
    SUMMARY_INSTRUCTION = "为这段摘要生成表示以用于检索相关文章："
    QUERY_INSTRUCTION = "为这个问题生成表示以用于检索相关文章："
    ANSWER_INSTRUCTION = "为这个答案生成表示以用于检索相关文章："

    INSTRUCTION_DICT = {
        "content_query": QUERY_INSTRUCTION,
        "content_answer": ANSWER_INSTRUCTION,
        "content_summary": SUMMARY_INSTRUCTION,
        "default": "为这个句子生成表示以用于检索相关文章："
    }

    # 常用路径初始化
    PROJECT_PATH = os.path.join(current_dir, "..")
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    DOCS_PATH = os.path.join(PROJECT_PATH, "docs")
    MODEL_PATH = os.path.join(PROJECT_PATH, "model_output")

    M3E_BASE_MODEL_PATH = os.environ.get("M3E_EMBEDDINGS_MODEL_PATH")
    M3E_ADAPTER_FINETUNE_MODEL_PATH = os.path.join(MODEL_PATH, "adapter-m3e-base")
    M3E_FINETUNE_MODEL_PATH = os.environ.get("M3E_FINETUNE_MODEL_PATH", os.path.join(MODEL_PATH, "m3e-base-finetune-v1.0"))

    BGE_BASE_MODEL_PATH = os.environ.get("BGE_EMBEDDINGS_MODEL_PATH")
    BGE_ADAPTER_FINETUNE_MODEL_PATH = os.path.join(MODEL_PATH, "adapter-bge-big-zh")
    BGE_FINETUNE_MODEL_PATH = os.environ.get("BGE_FINETUNE_MODEL_PATH", os.path.join(MODEL_PATH, "bge-large-zh-finetune-v1.0"))

    EVALUATE_MODEL_PATH = os.environ.get("EVALUATE_MODEL_PATH")
    RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH")

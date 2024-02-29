import __init__  # noqa
import os  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # noqa

from tools.logger import getLogger
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
from llama_index.schema import TextNode
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index import ServiceContext, VectorStoreIndex
from tools.utils import GLOBAL


logger = getLogger()


def get_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llama_index.llms import HuggingFaceLLM
    import torch

    evaluate_model_path = GLOBAL.EVALUATE_MODEL_PATH
    model = AutoModelForCausalLM.from_pretrained(evaluate_model_path, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(evaluate_model_path, trust_remote_code=True)

    _llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
    del model
    torch.cuda.empty_cache()
    return _llm


llm = get_llm()


class EvaluateLlamaIndexNew():
    @classmethod
    def retrieve_and_evaluate(cls, retriever, query_id, query, relevant_docs):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        return {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }

    @classmethod
    def evaluate(cls, train_params, dataset: EmbeddingQAFinetuneDataset, embed_model, top_k=5, verbose=False):
        logger.info(f"----------------evaluate begin, train_params({train_params}) top_k={top_k}----------------")
        corpus = dataset.corpus
        queries = dataset.queries
        relevant_docs = dataset.relevant_docs

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
        index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
        retriever = index.as_retriever(similarity_top_k=top_k)

        eval_results = []
        handle_count = 0
        total_count = len(queries.items())
        last_progress = 0
        hit_rate_count = 0

        for query_id, query in queries.items():
            retrieved_nodes = retriever.retrieve(query)
            retrieved_ids = [node.node.node_id for node in retrieved_nodes]
            expected_id = relevant_docs[query_id][0]
            is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

            eval_result = cls.retrieve_and_evaluate(retriever, query_id, query, relevant_docs)
            eval_results.append(eval_result)

            if is_hit:
                hit_rate_count += 1

            # 控制进度日志，百分比有变化且是5的倍数输出
            handle_count += 1
            current_progress = handle_count * 100 // total_count
            if last_progress != current_progress and (current_progress % 5 == 0):
                hit_rate = hit_rate_count * 100 // handle_count
                log_content = f'evaluate(top_k={top_k})处理进度：{handle_count}/{total_count},百分比：{current_progress}%,hit_rate:{hit_rate}%'
                logger.info(log_content)
                print(log_content)
                last_progress = current_progress

        hit_rate = hit_rate_count * 100 // handle_count
        logger.info(f"----------------evaluate finished, train_params({train_params}) top_k={top_k},hit_rate:{hit_rate}%")
        return eval_results

    @classmethod
    def evaluate_st(cls, train_params, dataset: EmbeddingQAFinetuneDataset, model_id, name):
        corpus = dataset.corpus
        queries = dataset.queries
        relevant_docs = dataset.relevant_docs

        evaluator = InformationRetrievalEvaluator(
            queries, corpus, relevant_docs, name=name
        )
        model = SentenceTransformer(model_id)
        output_path = "results/"
        Path(output_path).mkdir(exist_ok=True, parents=True)
        return evaluator(model, output_path=output_path)

    @classmethod
    def eval_model(cls, train_params, model_base_path, finetune_model_path, json_file_name, evaluate_base=True):
        if not os.path.isabs(json_file_name):
            json_file_name = f"{GLOBAL.DATA_PATH}/{json_file_name}"
        val_dataset = EmbeddingQAFinetuneDataset.from_json(json_file_name)

        if evaluate_base:
            model_base_name = f"local:{model_base_path}"
            bge_base_val_results = cls.evaluate(train_params, val_dataset, model_base_name)
            df_bge_base = pd.DataFrame(bge_base_val_results)
            hit_rate_bge_base = df_bge_base["is_hit"].mean()
            # self.evaluate_st(train_params, val_dataset, model_base_name, name="base")
        else:
            hit_rate_bge_base = "unknown"

        model_fintune_name = f"local:{finetune_model_path}"
        bge_finetune_val_results = cls.evaluate(train_params, val_dataset, model_fintune_name)
        df_bge_finetune = pd.DataFrame(bge_finetune_val_results)
        hit_rate_bge_finetune = df_bge_finetune["is_hit"].mean()
        # self.evaluate_st(train_params, val_dataset, model_fintune_name, name="finetune")

        hit_rate_str = f"hit_rate_origin:{hit_rate_bge_base},hit_rate_finetune:{hit_rate_bge_finetune}"
        logger.warn(
            f"eval_model finished, train_params:{train_params},finetune_model_path:{finetune_model_path},json_file_name={json_file_name},result as follows:\n{hit_rate_str}")

        if evaluate_base:
            df_bge_base["model"] = "base"
            df_bge_finetune["model"] = "finetune"

            df_all = pd.concat([df_bge_base, df_bge_finetune])
            model_grouped_mean = df_all.groupby("model").mean("is_hit")

            # Convert entire df_all DataFrame to a string for logging
            # df_all_str = df_all.to_string()
            # logger.info(f"Full df_all results:\n{df_all_str}")

            model_grouped_mean_str = model_grouped_mean.to_string()
            logger.info(
                f"eval_model finished, train_params:{train_params},finetune_model_path:{finetune_model_path},json_file_name={json_file_name},Grouped mean by model:\n{model_grouped_mean_str}")

        # df_st_bge = pd.read_csv(
        #     "results/Information-Retrieval_evaluation_base_results.csv"
        # )
        # df_st_finetuned = pd.read_csv(
        #     "results/Information-Retrieval_evaluation_finetune_results.csv"
        # )

        # df_st_bge["model"] = "base"
        # df_st_finetuned["model"] = "finetune"
        # df_st_all = pd.concat([df_st_bge, df_st_finetuned])
        # df_st_all = df_st_all.set_index("model")

    @classmethod
    def evaluate_m3e(cls, train_params, finetune_model_path, json_file_name, evaluate_base=True):
        model_base_path = f"{GLOBAL.M3E_BASE_MODEL_PATH}"
        cls.eval_model(train_params, model_base_path, finetune_model_path, json_file_name, evaluate_base)

    @classmethod
    def evaluate_bge(cls, train_params, finetune_model_path, json_file_name, evaluate_base=True):
        model_base_path = f"{GLOBAL.BGE_BASE_MODEL_PATH}"
        cls.eval_model(train_params, model_base_path, finetune_model_path, json_file_name, evaluate_base)


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    # EvaluateLlamaIndexNew.evaluate_m3e()

    # json_file_name = f"{GLOBAL.DATA_PATH}/datasets_qa_v1.0.json"

    # json_file_name = f"{GLOBAL.DATA_PATH}/datasets_v2.0_val.json"  base:0.8  finetune:0.9

    # json_file_name = f"{GLOBAL.DATA_PATH}/datasets_v1.0.json"

    # json_file_name = f"{GLOBAL.DATA_PATH}/datasets_v2.0.json"  # 旧 base:0.012506  finetune:0.019010

    json_file_name = f"{GLOBAL.DATA_PATH}/datasets_v2.2.json"  # 旧 base:0.012506  finetune:0.019010

    # EvaluateLlamaIndexNew.evaluate_m3e("m3e-base", GLOBAL.M3E_FINETUNE_MODEL_PATH, json_file_name)

    EvaluateLlamaIndexNew.evaluate_bge(
        "bge-large", "/alidata/models/BAAI/bge-large-zh-v1.5-finetune-v2.0-train/datasets_datasets_v2.2_epochs_5_batch_5_lr_1e-05_test_0.2",
        json_file_name, evaluate_base=False)

    pass

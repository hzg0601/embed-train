import __init__  # noqa
import os  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # noqa

import shutil
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from tools.utils import GLOBAL
from uniem.finetuner import FineTuner, ModelType
from evaluate.evaluate_model import EvaluateModel, EvaluateMethodType
from tools.logger import getLogger


logger = getLogger()


class UniemTrainModel(EvaluateModel):
    base_model_path = None
    finetune_model_path = None
    base_model_name = None
    model_type = None

    test_sizes = [0.2]
    lrs = [3e-5]
    epochs = [5]
    batch_sizes = [16]

    def __init__(self):
        super().__init__()

    def copy_and_delete(self, source_dir, target_dir):
        for file in os.listdir(source_dir):
            if not os.path.isdir(os.path.join(source_dir, file)):
                shutil.copy(os.path.join(source_dir, file), target_dir)

        shutil.rmtree(source_dir)

    def evaluate_model(self, train_params, finetune_model_path, json_file_name="",
                       evaluate_method_type: EvaluateMethodType = EvaluateMethodType.db_vector, evaluate_base=True):
        pass

    def get_finetune_model_path(self, epochs, batch_size, lr, test_size, files_list):
        # 将列表转换为字符串，元素之间用_分隔
        datasets_str = "_".join(files_list).replace(".csv", "").replace(".jsonl", "")
        output_sub_dir = f"datasets_{datasets_str}_epochs_{epochs}_batch_{batch_size}_lr_{lr}_test_{test_size}"
        finetune_model_path = os.path.join(self.finetune_model_path, output_sub_dir)
        return finetune_model_path

    def train_csv(
        self, epochs=5, batch_size=16, lr=3e-5, test_size=0.2, csv_files_list=None
    ):
        if csv_files_list is None:
            raise Exception("csv_files_list is None in train_csv")

        # 生成微调模型的输出路径
        finetune_model_path = self.get_finetune_model_path(
            epochs, batch_size, lr, test_size, csv_files_list
        )

        # 将列表转换为字符串，元素之间用逗号分隔
        csv_files_str = ",".join(csv_files_list)

        train_params = f"method:uniem,model:{self.base_model_name},datasets:[{csv_files_str}],epochs={epochs},batch_size={batch_size},lr={lr},test_size={test_size}"
        logger.info(
            f"train_csv train start,train_params:{train_params},finetune_model_path:{finetune_model_path}"
        )

        if len(csv_files_list) > 1:
            datasets_parts = []
            for file in csv_files_list:
                file_path = f"{GLOBAL.DATA_PATH}/{file}"
                dataset_part = load_dataset("csv", data_files=file_path)
                if "train" in dataset_part:
                    datasets_parts.append(dataset_part["train"])

            # 检测数据集是否有效
            if not datasets_parts:
                raise ValueError("No 'train' datasets found in the specified files.")

            # 合并数据集
            combined_dataset = concatenate_datasets(datasets_parts)

            # 分割数据
            split_dataset = combined_dataset.train_test_split(test_size=test_size)

            # 创建新的dataset字典，把'test'改为'validation'
            dataset = {
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            }

        else:  # 对于单个csv文件(备注：单个csv的流程在上述多csv文件处理上应该仍然是有效的，单独处理仅为了简化流程)
            file_path = f"{GLOBAL.DATA_PATH}/{csv_files_list[0]}"
            dataset = load_dataset("csv", data_files=file_path)

            if test_size is not None and test_size > 0.01:
                # 分割数据
                split_dataset = dataset["train"].train_test_split(test_size=test_size)

                # 创建新的dataset字典，把'test'改为'validation'
                dataset = {
                    "train": split_dataset["train"],
                    "validation": split_dataset["test"],
                }

        # 指定训练的模型
        finetuner = FineTuner.from_pretrained(
            self.base_model_path, dataset=dataset, model_type=self.model_type
        )
        _ = finetuner.run(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            log_with=["tensorboard"],
            output_dir=finetune_model_path,
        )

        # 移动 model子文件夹到上一层目录中
        self.copy_and_delete(f"{finetune_model_path}/model", finetune_model_path)

        logger.info(
            f"train_csv train finished and start evaluate,train_params:{train_params},finetune_model_path:{finetune_model_path}"
        )

        # 评估训练后的模型
        self.evaluate_model(train_params, finetune_model_path)

        logger.info(
            f"train_csv evaluate finished,train_params:{train_params},finetune_model_path:{finetune_model_path}"
        )

    def __load_jsonl_dataset(self, file_path):
        # 读取JSONL文件到DataFrame
        df = pd.read_json(file_path, lines=True)

        # 重命名列
        df = df.rename(columns={"query": "text", "pos": "text_pos"})

        # 移除不需要的列
        df = df.drop(columns="neg")

        # 检查'text_pos'字段是否为列表，如果是，则去除列表
        if isinstance(df.iloc[0]["text_pos"], list):
            df["text_pos"] = df["text_pos"].apply(lambda x: x[0] if x else None)

        # 将 DataFrame 转换为记录列表
        dataset = df.to_dict("records")
        return dataset

    def train_jsonl(
        self, epochs=3, batch_size=16, lr=3e-5, test_size=0.2, jsonl_files_list=None, json_file_name="",
        evaluate_method_type: EvaluateMethodType = EvaluateMethodType.db_vector, evaluate_base=True
    ):
        if jsonl_files_list is None:
            raise Exception("jsonl_files_list is None in train_jsonl")

        # 生成微调模型的输出路径
        finetune_model_path = self.get_finetune_model_path(
            epochs, batch_size, lr, test_size, jsonl_files_list
        )

        # 将列表转换为字符串，元素之间用逗号分隔
        jsonl_files_str = ",".join(jsonl_files_list)

        train_params = f"method:uniem,model:{self.base_model_name},datasets:[{jsonl_files_str}],epochs={epochs},batch_size={batch_size},lr={lr},test_size={test_size}"
        logger.info(
            f"train_jsonl train start,train_params:{train_params},finetune_model_path:{finetune_model_path}"
        )

        if len(jsonl_files_list) > 1:
            dataset = []
            for file in jsonl_files_list:
                file_path = f"{GLOBAL.DATA_PATH}/{file}"
                dataset_part = self.__load_jsonl_dataset(file_path)
                dataset.extend(dataset_part)  # use extend to add elements to the list

            # 检测数据集是否有效
            if not dataset:
                raise ValueError("No datasets found in the specified files.")
        else:
            # 读取 jsonl 文件
            file_path = f"{GLOBAL.DATA_PATH}/{jsonl_files_list[0]}"
            dataset = self.__load_jsonl_dataset(file_path)

        # 这里是所有数据集合并以后，再按比例拆分成训练集和验证集
        if test_size is not None and test_size > 0.01:
            # 分割数据
            train_records, validation_records = train_test_split(
                dataset, test_size=test_size
            )

            # 创建新的 dataset 字典
            dataset = {"train": train_records, "validation": validation_records}

        # 指定训练的模型
        finetuner = FineTuner.from_pretrained(
            self.base_model_path, dataset=dataset, model_type=self.model_type
        )
        _ = finetuner.run(
            epochs=epochs, batch_size=batch_size, lr=lr, output_dir=finetune_model_path
        )

        # 移动 model子文件夹到上一层目录中
        self.copy_and_delete(f"{finetune_model_path}/model", finetune_model_path)

        logger.info(
            f"train_jsonl train finished and start evaluate,train_params:{train_params},finetune_model_path:{finetune_model_path}"
        )

        # 评估训练后的模型
        self.evaluate_model(train_params, finetune_model_path, json_file_name, evaluate_method_type, evaluate_base)

        logger.info(
            f"train_jsonl evaluate finished,train_params:{train_params},finetune_model_path:{finetune_model_path}"
        )

    # 批量训练
    def batch_train_csv_trouble(self):
        # csv_files_lists = [['datasets_v1.0.csv'],['datasets_v1.1.csv'], ['datasets_v2.0.csv'], ['datasets_v1.1.csv', 'datasets_v2.0.csv']]
        csv_files_lists = [
            ["datasets_v1.1.csv", "datasets_v2.0.csv"],
        ]
        test_sizes = self.test_sizes
        lrs = self.lrs
        epochs = self.epochs
        batch_sizes = self.batch_sizes
        for csv_files_list in csv_files_lists:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    for lr in lrs:
                        for test_size in test_sizes:
                            self.train_csv(
                                epochs=epoch,
                                batch_size=batch_size,
                                lr=lr,
                                test_size=test_size,
                                csv_files_list=csv_files_list,
                            )

    # 批量训练
    def batch_train_csv_qa(self):
        csv_files_lists = [["datasets_qa_v1.0.csv"]]
        test_sizes = self.test_sizes
        lrs = self.lrs
        epochs = self.epochs
        batch_sizes = self.batch_sizes
        for csv_files_list in csv_files_lists:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    for lr in lrs:
                        for test_size in test_sizes:
                            self.train_csv(
                                epochs=epoch,
                                batch_size=batch_size,
                                lr=lr,
                                test_size=test_size,
                                csv_files_list=csv_files_list,
                            )

    # 批量训练
    def batch_train_jsonl_trouble(self):
        # jsonl_files_lists = [['datasets_qa_v1.0.jsonl'],['datasets_1.0.jsonl'], ['datasets_v2.0.jsonl']]
        # jsonl_files_lists = [['datasets_v1.0.jsonl'], ['datasets_v2.0.jsonl'], ['datasets_v1.1.jsonl']]
        # jsonl_files_lists = [['datasets_qa_v1.0.jsonl', 'datasets_v2.0.jsonl', 'datasets_v1.0.jsonl'],
        #                      ['datasets_qa_v1.0.jsonl', 'datasets_v2.0.jsonl', 'datasets_v1.1.jsonl'],
        #                      ['datasets_qa_v1.0.jsonl', 'datasets_v1.0.jsonl'],
        #                      ['datasets_qa_v1.0.jsonl', 'datasets_v2.0.jsonl'],
        #                      ['datasets_v2.0.jsonl', 'datasets_v1.0.jsonl'],
        #                      ['datasets_v2.0.jsonl', 'datasets_v1.1.jsonl']
        #                      ]
        jsonl_files_lists = [["datasets_v2.2.jsonl"]]

        # 用llama-index进行评估
        json_file_name = "datasets_v2.2.json"
        evaluate_method_type = EvaluateMethodType.llama_index

        test_sizes = self.test_sizes
        lrs = self.lrs
        epochs = [30]
        batch_sizes = self.batch_sizes
        for jsonl_files_list in jsonl_files_lists:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    for lr in lrs:
                        for test_size in test_sizes:
                            self.train_jsonl(
                                epochs=epoch,
                                batch_size=batch_size,
                                lr=lr,
                                test_size=test_size,
                                jsonl_files_list=jsonl_files_list,
                                json_file_name=json_file_name,
                                evaluate_method_type=evaluate_method_type,
                                evaluate_base=False
                            )

    # 批量训练(问答对训练)
    def batch_train_jsonl_qa(self):
        jsonl_files_lists = [["datasets_qa_v1.0.jsonl"]]

        # 用llama-index进行评估
        json_file_name = "datasets_qa_v1.0.jsonl"
        evaluate_method_type = EvaluateMethodType.llama_index

        test_sizes = self.test_sizes
        lrs = self.lrs
        epochs = self.epochs
        batch_sizes = self.batch_sizes
        for jsonl_files_list in jsonl_files_lists:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    for lr in lrs:
                        for test_size in test_sizes:
                            self.train_jsonl(
                                epochs=epoch,
                                batch_size=batch_size,
                                lr=lr,
                                test_size=test_size,
                                jsonl_files_list=jsonl_files_list,
                                json_file_name=json_file_name,
                                evaluate_method_type=evaluate_method_type
                            )


class UniemTrainM3E(UniemTrainModel):
    def __init__(self):
        self.base_model_path = GLOBAL.M3E_BASE_MODEL_PATH
        self.finetune_model_path = GLOBAL.M3E_FINETUNE_MODEL_PATH
        self.base_model_name = "m3e-base"
        self.model_type = ModelType.uniem

        self.lrs = [3e-5]
        self.batch_sizes = [16]

        super().__init__()

    def evaluate_model(self, train_params, finetune_model_path, json_file_name="", evaluate_method_type: EvaluateMethodType = EvaluateMethodType.db_vector, evaluate_base=True):
        self.evaluate_m3e(train_params, finetune_model_path, json_file_name, evaluate_method_type, evaluate_base)


class UniemTrainBGE(UniemTrainModel):
    def __init__(self):
        self.base_model_path = GLOBAL.BGE_BASE_MODEL_PATH
        self.finetune_model_path = GLOBAL.BGE_FINETUNE_MODEL_PATH
        self.base_model_name = "bge-large"
        self.model_type = ModelType.sentence_transformers

        self.lrs = [1e-5]
        self.batch_sizes = [5]

        super().__init__()

    def evaluate_model(self, train_params, finetune_model_path, json_file_name="", evaluate_method_type: EvaluateMethodType = EvaluateMethodType.db_vector, evaluate_base=True):
        self.evaluate_bge(train_params, finetune_model_path, json_file_name, evaluate_method_type, evaluate_base)


if __name__ == "__main__":  # sourcery skip: raise-specific-error
    from typing import List

    train_tasks: List[UniemTrainModel] = []

    # uniem_train_m3e = UniemTrainM3E()
    # train_tasks.append(uniem_train_m3e)

    uniem_train_bge = UniemTrainBGE()
    train_tasks.append(uniem_train_bge)

    for train_task in train_tasks:
        # train_task.batch_train_csv_trouble()
        # train_task.batch_train_csv_qa()

        train_task.batch_train_jsonl_trouble()
        # train_task.batch_train_jsonl_qa()

    pass

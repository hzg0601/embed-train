import __init__
import csv
import json
import uuid
import random
from tools.utils import GLOBAL

from tools.logger import getLogger

logger = getLogger()


class DataSetConvert:

    # 你需要定义generate_neg函数来生成neg字段的值。
    @classmethod
    def generate_neg(self, pos_text, all_samples):
        # 随机选择一个样本作为负样本，且确保它与正样本不同
        neg_sample = pos_text
        while neg_sample == pos_text:
            neg_sample = random.choice(all_samples)
        return neg_sample

    @classmethod
    def csv_2_jsonl(self, csv_file_path, jsonl_file_path, pos_neg_to_list=True):
        # 第一次读取CSV文件来构建所有可能text_pos的列表
        all_samples = []
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            all_samples = [row['text_pos'] for row in reader]

        # 打开CSV文件和JSONL文件准备读写
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file, \
                open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:

            # 创建一个CSV读取器
            reader = csv.DictReader(csv_file)

            # 遍历CSV文件的每一行
            for row in reader:
                # 提取text和text_pos字段
                text = row['text']
                text_pos = row['text_pos']

                # 生成neg字段
                neg = self.generate_neg(text_pos, all_samples)

                # 构造想要的字典结构
                if pos_neg_to_list:
                    json_record = {"query": text, "pos": [text_pos], "neg": [neg]}
                else:
                    json_record = {"query": text, "pos": text_pos, "neg": neg}

                # 将合并后的JSON数据转换为格式化的字符串，提高可读性
                readable_json = json.dumps(json_record, ensure_ascii=False)

                # 将字典转换为JSON字符串并写入文件
                jsonl_file.write(readable_json + '\n')

        # 完成转换
        logger.info(f"csv_2_jsonl finished,jsonl_file_path:{jsonl_file_path}")

    @classmethod
    def csv_2_json(self, csv_file_path, json_file_path, instruction_flag=False):
        # Initialize dictionaries
        nodes_dict, queries_dict, relevant_docs = {}, {}, {}

        # 读取CSV文件并进行处理
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                # Generate unique identifiers
                question_id = str(uuid.uuid4())
                doc_id = str(uuid.uuid4())

                # Retrieve document and query
                doc = row['text_pos'].replace('\n', '\\n')
                query = row['text']
                if instruction_flag:
                    query = GLOBAL.INSTRUCTION_DICT["default"] + query

                # Populate dictionaries
                nodes_dict[doc_id] = doc
                queries_dict[question_id] = query
                relevant_docs[question_id] = [doc_id]  # Ensure doc_id is in a list as required

        # Prepare the final data structure
        final_data = {
            "queries": queries_dict,
            "corpus": nodes_dict,
            "relevant_docs": relevant_docs
        }

        # Write to json file
        with open(json_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(final_data, outfile, ensure_ascii=False, indent=4)

        logger.info(f'csv_2_json finished,json_file_path：{json_file_path}')

    @classmethod
    def jsonl_2_json(self, jsonl_file_path, json_file_path, instruction_flag=False):
        # Initialize dictionaries
        nodes_dict, queries_dict, relevant_docs = {}, {}, {}

        with open(jsonl_file_path, "r") as f:
            for line in f:
                line = json.loads(line)
                if not line.get("pos"):  # Check if 'pos' key exists and has content
                    continue

                # Generate unique identifiers
                question_id = str(uuid.uuid4())
                doc_id = str(uuid.uuid4())

                # Retrieve document and query
                doc = line["pos"][0]
                query = line["query"]
                if instruction_flag:
                    query = GLOBAL.INSTRUCTION_DICT["default"] + query

                # Populate dictionaries
                nodes_dict[doc_id] = doc
                queries_dict[question_id] = query
                relevant_docs[question_id] = [doc_id]  # Ensure doc_id is in a list as required

        # Prepare the final data structure
        final_data = {
            "queries": queries_dict,
            "corpus": nodes_dict,
            "relevant_docs": relevant_docs
        }

        # Write to json file
        with open(json_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(final_data, outfile, ensure_ascii=False, indent=4)

        logger.info(f"jsonl_2_json complete. Output saved to {json_file_path}")

    @classmethod
    def export_to_csv(self, csv_file_path, datasets):
        with open(csv_file_path, mode='w', encoding='utf-8', newline='') as csv_file:
            # 创建CSV writer对象
            fieldnames = ['text', 'text_pos']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # 写入CSV表头
            writer.writeheader()

            # 打印或处理结果
            for kb_source, data in datasets.items():
                kb_id = data['kb_id']
                paragraphs = data["paragraphs"]

                for paragraph in paragraphs:
                    kb_question = paragraph['kb_question']
                    kb_answer = paragraph['kb_answer']

                    # 将JSON数据行转换为CSV所需格式
                    row = {
                        'text': kb_question,
                        'text_pos': kb_answer
                    }

                    # 写入CSV文件
                    writer.writerow(row)

        logger.info(f"export_to_csv finished,csv_file_path:{csv_file_path}")


def test_csv_2_jsonl():
    # CSV文件路径
    csv_file_path = '/alidata/GitLab/train-embed-model/data/datasets_v2.2.csv'

    # 输出的JSONL文件路径
    jsonl_file_path = '/alidata/GitLab/train-embed-model/data/datasets_v2.2.jsonl'

    DataSetConvert.csv_2_jsonl(csv_file_path, jsonl_file_path)


def test_jsonl_2_json():
    # JSONL文件路径
    jsonl_file_path = '/alidata/GitLab/train-embed-model/data/datasets_v2.0.jsonl'

    # 输出的JSONL文件路径
    json_file_path = '/alidata/GitLab/train-embed-model/data/datasets_v2.0.json'

    DataSetConvert.jsonl_2_json(jsonl_file_path, json_file_path, False)


def test_csv_2_json():
    # CSV文件路径
    csv_file_path = '/alidata/GitLab/train-embed-model/data/datasets_v2.2.csv'
    
    # 输出的JSON文件路径
    json_file_path = '/alidata/GitLab/train-embed-model/data/datasets_v2.2.json'

    DataSetConvert.csv_2_json(csv_file_path, json_file_path)


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    test_csv_2_jsonl()

    # test_csv_2_json()

    # test_datasets_2_csv()

    # test_jsonl_2_json()
    pass

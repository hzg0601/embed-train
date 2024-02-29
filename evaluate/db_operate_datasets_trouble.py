import __init__
import tools.utils

from generate_qa.generate_trouble import GenerateTrouble
from evaluate.db_operate_datasets_base import DBOperateDatasets
from tools.logger import getLogger

logger = getLogger()


# 规范数据集（隐患数据），访问的是datasets表
class DBOperateDatasetsTrouble(DBOperateDatasets):
    # 数据集的表名
    DATASETS_TABLE_NAME = "datasets"

    def process_paragraph(self, paragraph):
        train_data_generate = GenerateTrouble()
        question = paragraph["text"].strip()
        return train_data_generate.generateData(question)


if __name__ == '__main__':  # sourcery skip: raise-specific-error

    dbOperateDatasetsTrouble = DBOperateDatasetsTrouble()
    dbOperateDatasetsTrouble.generate_all_datasets(kb_source='GB 50720-2011 建设工程施工现场消防安全技术规范.docx')

    pass

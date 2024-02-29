import __init__
from tools.utils import GLOBAL

from generate_qa.generate_qa import GenerateQA
from generate_qa.dataset_convert import DataSetConvert
from evaluate.db_operate_datasets_base import DBOperateDatasets
from tools.logger import getLogger

logger = getLogger()


# 规范数据集（问答对），访问的是datasets_qa表
class DBOperateDatasetsQA(DBOperateDatasets):
    # 数据集的表名
    DATASETS_TABLE_NAME = "datasets_qa"

    def process_paragraph(self, paragraph):
        train_data_generate = GenerateQA()
        question = paragraph["text"].strip()
        return train_data_generate.generateData(question)
    
def generate_all_datasets(kb_source=None):
    dbOperateDatasetsQA = DBOperateDatasetsQA()

    # 单本规范问答对生成，支持继续生成
    # dbOperateDatasetsQA.generate_all_datasets(kb_source='GB 50720-2011 建设工程施工现场消防安全技术规范.docx', continue_code='5.3.18', force_update=True)
    
    # 单本规范问答对生成，全新更新
    # dbOperateDatasetsQA.generate_all_datasets(kb_source='JGJ160-2016 施工现场机械设备检查技术规范.docx')
    
    # 没有生成过的规范，全部生成
    dbOperateDatasetsQA.generate_all_datasets()

# 导出数据集
def export_all_datasets(kb_source=None):
    dbOperateDatasetsQA = DBOperateDatasetsQA()
    results = dbOperateDatasetsQA.get_datasets_by_source(kb_source)
    
    csv_file_path = f"{GLOBAL.DATA_PATH}/datasets_qa.csv"
    DataSetConvert().export_to_csv(csv_file_path,results)


if __name__ == '__main__':  # sourcery skip: raise-specific-error

    # generate_all_datasets()
    
    export_all_datasets()
    pass

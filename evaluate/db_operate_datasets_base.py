import __init__
import tools.utils

import psycopg2
from evaluate.db_operate_specification import DBOperateSpecification

from evaluate.db_connect_pg import DBConnectPG

from tools.logger import getLogger

import concurrent.futures

logger = getLogger()


# 数据集操作父类，不要直接创建
class DBOperateDatasets(DBConnectPG):
    # 数据集的表名
    DATASETS_TABLE_NAME = ""
    
    # 生成数据集时的最大线程数，如果要使用单线程，设置为1
    MAX_THREADS = 10

    def __init__(self):  # sourcery skip: raise-specific-error
        super().__init__("AI_")  # 暂时放在AI工地巡检宝数据库中

    # 根据规范名称和规范Id双重查询，有一个有记录，则不允许 (不同用户允许)
    def check_dataset_exists(self, insert_kb_source, insert_kb_id):
        # 创建一个查询游标对象，用于执行SQL查询语句
        cursor_query = self.connection.cursor()
        try:
            sql_query = f"SELECT count(*) FROM {self.DATASETS_TABLE_NAME} WHERE (kb_id = '{insert_kb_id}' or kb_source = '{insert_kb_source}');"
            cursor_query.execute(sql_query)
            rows_query = cursor_query.fetchall()

            # 返回结果不是一个字段的话，说明出错了，先按数据库存在返回，避免后面的执行
            check_result = len(rows_query) == 1
            if not check_result:
                logger.error(f"check_dataset_exists执行错误，kb_id:{insert_kb_id},kb_source:{insert_kb_source}")
                return True

            return rows_query[0][0] > 0
        finally:
            # 关闭游标
            cursor_query.close()

    # 删除数据集，根据规范Id匹配删除
    def delete_datasets(self, insert_kb_id):
        # 创建一个查询游标对象，用于执行SQL查询语句
        cursor_update = self.connection.cursor()
        try:
            # 使用参数化查询，避免SQL注入
            sql_update = F"DELETE FROM {self.DATASETS_TABLE_NAME} WHERE kb_id = %s;"

            cursor_update.execute(sql_update, (insert_kb_id))
            self.connection.commit()
            cursor_update.close()
        finally:
            # 关闭游标
            cursor_update.close()

    def __inner_insert_datasets(self, insert_kb_source, insert_kb_id, insert_kb_answer, results):
        for insert_kb_question in results:
            # 使用参数化查询，避免SQL注入
            sql_insert = f"INSERT INTO {self.DATASETS_TABLE_NAME} (kb_id, kb_source, kb_question, kb_answer) VALUES(%s, %s, %s, %s);"

            cursor_insert = self.connection.cursor()
            cursor_insert.execute(
                sql_insert, (insert_kb_id, insert_kb_source, insert_kb_question, insert_kb_answer))
            self.connection.commit()
            cursor_insert.close()

        print(f'insert_datasets进度({insert_kb_source})：{self.handle_count}/{self.total_count},{self.handle_count * 100 // self.total_count}%')

    def process_paragraph(self, paragraph):
        raise Exception("Not Implemented process_paragraph")

    # 插入数据集
    def generate_all_datasets(self, kb_source=None, continue_code=None, force_update=False, force_delete=False):
        dbOperateSpecification = DBOperateSpecification()

        results = dbOperateSpecification.fetch_paragraphs_by_source(kb_source)

        # 打印或处理结果
        for kb_source, data in results.items():
            kb_id = data['kb_id']
            paragraphs = data["paragraphs"]
            self.insert_datasets(kb_source, kb_id, paragraphs, continue_code, force_update, force_delete)

    # 插入数据集
    def insert_datasets(self, insert_kb_source, insert_kb_id, paragraphs, continue_code=None, force_update=False, force_delete=False):
        try:
            self._connect_db()

            # 判断数据库中是否已经存在记录
            kb_exists = self.check_dataset_exists(insert_kb_source, insert_kb_id)
            if kb_exists and not force_update:
                logger.error(f"数据库中已经存在该数据集，kb_id:{insert_kb_id},kb_source:{insert_kb_source}")
                return

            # 数据库已经存在，且要求删除时，先删除数据库中的记录
            if kb_exists and force_update and force_delete:
                self.delete_datasets(insert_kb_source, insert_kb_id)

            check_continue_code = continue_code is not None and len(continue_code.strip()) > 0

            # 规范续传，节省生成内容
            start_index = 0   # 要截取的起始索引
            if (check_continue_code):
                for paragraph in paragraphs:
                    current_code = paragraph["prefix"]
                    if (current_code == continue_code):
                        start_index += 1  # 这里返回 + 1，避免重复生成，也就是说指定编号的数据集不会重新生成，如果有缺失，外部传参的时候，可以选上一个正确的
                        break

                    start_index += 1
                if (start_index > 0) and (start_index < len(paragraphs) - 1):
                    paragraphs = paragraphs[start_index:]  # 使用切片操作截取数据
                else:
                    logger.error(f"数据集继续生成逻辑错误，请检查序号，kb_id:{insert_kb_id},kb_source:{insert_kb_source}")
                    return

            # 处理结果
            self.handle_count = 0
            self.total_count = len(paragraphs)

            if (self.MAX_THREADS > 1):  # 使用多线程
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_THREADS) as executor:
                    future_to_paragraph = {executor.submit(self.process_paragraph, paragraph): paragraph for paragraph in paragraphs}

                    # 处理已完成的任务
                    for future in concurrent.futures.as_completed(future_to_paragraph):
                        self.handle_count += 1
                        paragraph = future_to_paragraph[future]
                        insert_kb_answer = paragraph["text"].strip()
                        results = future.result()

                        # 插入数据库的操作，类似前面示例的操作
                        self.__inner_insert_datasets(insert_kb_source, insert_kb_id, insert_kb_answer, results)

                    # 所有任务完成后，关闭线程池

            else:  # 使用单线程
                for paragraph in paragraphs:
                    self.handle_count += 1
                    insert_kb_answer = paragraph["text"].strip()
                    results = self.process_paragraph(paragraph)

                    # 插入数据库的操作，类似前面示例的操作
                    self.__inner_insert_datasets(insert_kb_source, insert_kb_id, insert_kb_answer, results)

            # 关闭游标和数据库连接
            self._disconnect_db()

        except psycopg2.Error as e:
            print("Error connecting to the database:", e)
            logger.error("Error connecting to the database:%s", e)
        except Exception as e:
            print("Error in insert_datasets:", e)
            logger.error("Error in insert_datasets:%s", e)
            
    def get_datasets_by_source(self, kb_source=None):
        try:
            self._connect_db()

            cursor = self.connection.cursor()

            # 查询所有知识库的kb_source及其相应的条文内容、前缀
            check_source_sql = f"and kb_source='{kb_source}'" if kb_source else ""
            sql = f"""
                SELECT kb_id, kb_source, kb_question, kb_answer 
                FROM {self.DATASETS_TABLE_NAME} where enabled=1 {check_source_sql} order by kb_id,id
            """
            cursor.execute(sql)
            records = cursor.fetchall()

            # 存储最终结果，此处使用kb_source作为键，但每个source包括其ID
            source_paragraphs = {}

            # 遍历查询结果，组织数据结构
            for kb_id, kb_source, kb_question, kb_answer in records:
                # 确保每个kb_source只被添加一次
                if kb_source not in source_paragraphs:
                    source_paragraphs[kb_source] = {
                        "kb_id": kb_id,
                        "paragraphs": []
                    }

                # 将条文信息作为列表项添加到相应的kb_source下
                paragraph = {"kb_question": kb_question, "kb_answer": kb_answer}
                source_paragraphs[kb_source]["paragraphs"].append(paragraph)

            # 关闭游标和数据库连接
            self._disconnect_db()

        except psycopg2.Error as e:
            print("Error connecting to the database in get_datasets_by_source:", e)
            logger.error("Error connecting to the database in get_datasets_by_source:%s", e)
        except Exception as e:
            print("Error in get_datasets_by_source:", e)
            logger.error("Error in get_datasets_by_source:%s", e)

        return source_paragraphs
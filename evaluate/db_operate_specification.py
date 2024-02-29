import os
import __init__
from tools.utils import GLOBAL
from tools.timer import Timer
import psycopg2
from evaluate.db_operate_base import DBOperateBase
from tools.logger import getLogger

logger = getLogger()


# 规范数据库操作类，访问的是eyeglasskb表(AI工地巡检宝)
class DBOperateSpecification(DBOperateBase):
    # m3e和bge模型微调的数据库向量字段名称（正式）
    # DB_NAME_PMS_M3E_VECTOR = 'pms_m3e_vector_content'
    # DB_NAME_PMS_BGE_LARGE_VECTOR = 'pms_bge_large_vector_content'

    # m3e和bge模型微调的数据库向量字段名称（训练使用）
    DB_NAME_PMS_M3E_VECTOR = 'pms_m3e_vector_content_train'
    DB_NAME_PMS_BGE_LARGE_VECTOR = 'pms_bge_large_vector_content_train'

    def __init__(self):  # sourcery skip: raise-specific-error
        super().__init__("AI_")

    # 单个向量的更新
    def update_vector(self, update_kb_id=None, content_field_name='kb_content', vector_field_name='m3e_vector_content', force_update=False, finetune_model_path=""):
        try:
            self._connect_db()

            # 是否只更新字段内容为null的数据，默认为True，以便能中断后继续执行；   如果需要强制更新所有，则only_update_field_is_null设置为False
            only_update_field_is_null = not force_update

            check_field_is_null_sql = f" and {vector_field_name} is null " if only_update_field_is_null else ""

            # 创建一个查询游标对象，用于执行SQL查询语句
            cursor_query = self.connection.cursor()

            # sql_query = f"SELECT id,{content_field_name} FROM eyeglasskb ORDER BY ID ASC LIMIT 1;"
            if update_kb_id is not None:
                sql_query = f"SELECT id,{content_field_name} FROM eyeglasskb where kb_id = '{update_kb_id}' and {content_field_name} is not null {check_field_is_null_sql} ORDER BY ID ASC;"
            else:
                sql_query = f"SELECT id,{content_field_name} FROM eyeglasskb where {content_field_name} is not null {check_field_is_null_sql} ORDER BY ID ASC;"

            cursor_query.execute(sql_query)
            rows_query = cursor_query.fetchall()

            # 处理查询结果
            handle_count = 0
            total_count = len(rows_query)
            last_progress = 0
            for row in rows_query:
                # row[1]为返回结果的第二个字段，即 content_field_name
                input_vector = self.to_vector(row[1], vector_field_name, False, finetune_model_path)
                if input_vector is None:
                    continue

                # 使用参数化查询，避免SQL注入
                sql_update = f"UPDATE eyeglasskb SET {vector_field_name} = %s WHERE id = %s;"
                cursor_update = self.connection.cursor()

                # row[0]为返回结果的第一个字段，即 id
                cursor_update.execute(sql_update, (input_vector, row[0]))
                self.connection.commit()
                cursor_update.close()

                # 控制进度日志，百分比有变化且是5的倍数输出
                handle_count += 1
                current_progress = handle_count * 100 // total_count
                if last_progress != current_progress and (current_progress % 5 == 0):
                    log_content = f'update_vector({vector_field_name})处理进度：{handle_count}/{total_count},百分比：{current_progress}%'
                    print(f"{Timer.current_time_str()}:{log_content}")
                    last_progress = current_progress

            # 关闭游标和数据库连接
            cursor_query.close()
            self._disconnect_db()

            return handle_count

        except psycopg2.Error as e:
            logger.error("Error connecting to the database:%s", e)
        except Exception as e:
            logger.error("Error in update_vector:%s", e)

        # 如果返回0，表示没有更新
        return 0

    # 数据库查询接口
    def kb_query(self, input_text, input_kb_id, compare_vector_field, compare_vector_operator,
                 input_similarity=0.8, input_limit=15, finetune_model_path=""):
        try:
            self._connect_db()

            # 创建一个游标对象，用于执行SQL语句
            cursor = self.connection.cursor()

            # 执行查询或其他操作
            input_vector = self.to_vector(input_text, compare_vector_field, True, finetune_model_path)
            if input_vector is None:
                return ""

            check_kb_id_sql = f" and a.kb_id='{input_kb_id}' " if len(input_kb_id) > 0 else ""

            # 检查规范分组
            check_kb_group_sql = ""

            # 检查规范分类
            check_kb_type_sql = ""

            # SQL语句中向量运算符说明：
            # <#>	negative inner product   负内积
            # <->	Euclidean distance       欧几里得距离
            # <=>	cosine distance          余弦距离

            # 注意，下面三个条件的SQL语句中，只有第3行的条件是不一样的，因此引入check_similarity_sql做统一优化
            vector_operator_value = f"a.{compare_vector_field} {compare_vector_operator} '{input_vector}'"
            check_similarity_sql = ""
            if compare_vector_operator == "<#>":
                # 做相似度判断
                check_similarity_sql = f" and {vector_operator_value} < -{input_similarity}"

            elif compare_vector_operator == "<->":
                # 这种写法是找到最匹配的记录，不做相似度判断
                check_similarity_sql = ""

            elif compare_vector_operator == "<=>":
                # 做相似度判断
                check_similarity_sql = f" and {vector_operator_value} <= 1 - {input_similarity}"

            sql = f"SELECT a.id,a.kb_id,a.kb_source,a.kb_content,a.kb_title,a.kb_sub_title,{vector_operator_value} as vector_difference,a.content_prefix FROM eyeglasskb as a \
                    INNER JOIN kbsources as b ON a.kb_id = b.kb_id where 1=1 {check_kb_id_sql} {check_kb_group_sql} {check_kb_type_sql} \
                    {check_similarity_sql} \
                    order by {vector_operator_value} LIMIT {input_limit};"
            cursor.execute(sql)
            rows = cursor.fetchall()

            # 关闭游标和数据库连接
            cursor.close()
            self._disconnect_db()

            return rows

        except psycopg2.Error as e:
            print("Error connecting to the database:", e)
        except Exception as e:
            print("Error in kb_query:", e)

    def fetch_paragraphs_by_source(self, kb_source=None):
        try:
            self._connect_db()

            cursor = self.connection.cursor()

            # 查询所有知识库的kb_id, kb_source及其相应的条文内容、前缀
            check_source_sql = f"and kb_source='{kb_source}'" if kb_source else ""
            sql = f"""
                SELECT kb_id, kb_source, content_prefix, kb_content 
                FROM eyeglasskb where 1=1 {check_source_sql} order by kb_id, id
            """
            cursor.execute(sql)
            records = cursor.fetchall()

            # 存储最终结果，此处使用kb_source作为键，但每个source包括其ID
            source_paragraphs = {}

            # 遍历查询结果，组织数据结构
            for kb_id, source, prefix, content in records:
                # 确保每个kb_source只被添加一次
                if source not in source_paragraphs:
                    source_paragraphs[source] = {
                        "kb_id": kb_id,
                        "paragraphs": []
                    }

                # 将条文信息作为列表项添加到相应的kb_source下
                paragraph = {"prefix": prefix, "text": content}
                source_paragraphs[source]["paragraphs"].append(paragraph)

            # 关闭游标和数据库连接
            self._disconnect_db()

        except psycopg2.Error as e:
            print("Error connecting to the database in fetch_paragraphs_by_source:", e)
            logger.error("Error connecting to the database in fetch_paragraphs_by_source:%s", e)
        except Exception as e:
            print("Error in fetch_paragraphs_by_source:", e)
            logger.error("Error in fetch_paragraphs_by_source:%s", e)

        return source_paragraphs

    # 创建向量索引
    # 注意，pgvector向量维数必须指定，创建索引才能成功，如果原先创建的时候没有指定维数，则需要删除字段重建，或者采用别的方法更新
    # 向量维数，根据不同的模型约定，比如，m3e-base是768维，bge-large是1024维，具体看各模型参数
    def create_vector_index(self, vector_field_name='m3e_vector_content'):
        try:
            self._connect_db()

            cursor = self.connection.cursor()

            # 检查并创建 L2 索引
            sql_check_l2 = f"SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_{vector_field_name}_l2';"
            cursor.execute(sql_check_l2)
            if not cursor.fetchone():
                sql_l2 = f"CREATE INDEX idx_{vector_field_name}_l2 ON public.eyeglasskb USING ivfflat ({vector_field_name} vector_l2_ops) WITH (lists = 100);"
                cursor.execute(sql_l2)

            # 检查并创建 IP 索引
            sql_check_ip = f"SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_{vector_field_name}_ip';"
            cursor.execute(sql_check_ip)
            if not cursor.fetchone():
                sql_ip = f"CREATE INDEX idx_{vector_field_name}_ip ON public.eyeglasskb USING ivfflat ({vector_field_name} vector_ip_ops) WITH (lists = 100);"
                cursor.execute(sql_ip)

            # 检查并创建余弦相似度（Cosine）索引
            sql_check_cosine = f"SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_{vector_field_name}_cosine';"
            cursor.execute(sql_check_cosine)
            if not cursor.fetchone():
                sql_cosine = f"CREATE INDEX idx_{vector_field_name}_cosine ON public.eyeglasskb USING ivfflat ({vector_field_name} vector_cosine_ops) WITH (lists = 100);"
                cursor.execute(sql_cosine)

            # 提交事务，使得之前执行的 SQL 生效
            self.connection.commit()
        except psycopg2.Error as e:
            logger.error("Error connecting to the database in create_vector_index:%s", e)
            if self.connection:
                self.connection.rollback()  # 如果出现错误，进行回滚
        except Exception as e:
            logger.error("Error in create_vector_index:%s", e)
            if self.connection:
                self.connection.rollback()  # 如果出现错误，进行回滚
        finally:
            # 无论成功还是失败都关闭游标和数据库连接
            if cursor:
                cursor.close()
            self._disconnect_db()


def update_pms_m3e_vector(finetune_model_path=""):
    dbOperateSpecification = DBOperateSpecification()

    # 更新pms_m3e-base向量字段
    handle_count = dbOperateSpecification.update_vector(
        None, 'kb_content', DBOperateSpecification.DB_NAME_PMS_M3E_VECTOR, True, finetune_model_path)

    # 更新pms_m3e-base向量字段(训练)
    handle_count = dbOperateSpecification.update_vector(None, 'kb_content', 'pms_m3e_vector_content_train', True, finetune_model_path)

    return handle_count


def update_pms_bge_large_vector(finetune_model_path=""):
    dbOperateSpecification = DBOperateSpecification()

    # 更新pms_bge_large_vector_content向量字段
    handle_count = dbOperateSpecification.update_vector(
        None, 'kb_content', DBOperateSpecification.DB_NAME_PMS_BGE_LARGE_VECTOR, True, finetune_model_path)

    return handle_count


def create_vector_index():
    dbOperateSpecification = DBOperateSpecification()
    vector_field_names = ["m3e_vector_content", "pms_m3e_vector_content",  "pms_m3e_vector_content_train",
                          "bge_large_vector_content", "pms_bge_large_vector_content", "pms_bge_large_vector_content_train"]
    for vector_field_name in vector_field_names:
        dbOperateSpecification.create_vector_index(vector_field_name)


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    finetune_model_path = f'{GLOBAL.M3E_FINETUNE_MODEL_PATH}/datasets_datasets_qa_epochs_5_batch_16_lr_3e-05_test_0.2'
    # update_pms_m3e_vector()

    finetune_model_path = f'{GLOBAL.BGE_FINETUNE_MODEL_PATH}/datasets_datasets_qa_epochs_5_batch_6_lr_1e-05_test_0.2'
    # update_pms_bge_large_vector(finetune_model_path)

    # create_vector_index()
    pass

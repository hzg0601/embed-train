import json
import os

import __init__

import psycopg2

# Postgresql连接
class DBConnectPG():

    # 定义数据库连接参数
    db_params = {}
    
    prefix = ""

    connection = None

    def __init__(self, config_prefix):  # sourcery skip: raise-specific-error
        super().__init__()
        
        self.prefix = config_prefix

        host_str = os.environ.get(self.prefix + "POSTGRES_DB_HOST")
        port_str = os.environ.get(self.prefix + "POSTGRES_DB_PORT")
        database_str = os.environ.get(self.prefix + "POSTGRES_DB_DATABASE")
        user_str = os.environ.get(self.prefix + "POSTGRES_DB_USER")
        password_str = os.environ.get(self.prefix + "POSTGRES_DB_PASSWORD")

        # 定义数据库连接参数
        self.db_params = {
            "host": host_str,
            "database": database_str,
            "user": user_str,
            "password": password_str,
            "port": port_str
        }

    def _connect_db(self):
        # 连接到数据库
        self.connection = psycopg2.connect(**self.db_params)

    def _disconnect_db(self):
        # 断开数据库连接
        self.connection.close()


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    pass

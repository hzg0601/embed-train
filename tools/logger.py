import os
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def singleton(class_):
    instances = {}
    def wrapper(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return wrapper

@singleton
class SingletonLogger:
    def __init__(self):
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 创建 TimedRotatingFileHandler
        unique_id = os.getenv('UNIQUE_ID', 'default')

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, "..", "log")
            
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            
        # 获取当前时间
        current_time = datetime.now()

        # 将时间格式化为字符串，一天一个日志文件，避免过大，也避免查找不方便
        current_date_str = current_time.strftime("%Y_%m_%d")
            
        log_file_name = os.path.join(log_dir,f'spec_{unique_id}_{current_date_str}.log')
            
        handler = TimedRotatingFileHandler(filename=log_file_name, when='D', interval=1, backupCount=0)
        #handler = TimedRotatingFileHandler(filename='/opt/spec-log/spec.log', when='D', interval=1, backupCount=0)
        handler.setLevel(logging.INFO)

        # 创建 formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s  ')
        handler.setFormatter(formatter)

        # 将 handler 添加到 logger 中
        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger

def getLogger():
    logger_instance = SingletonLogger().get_logger()
    return logger_instance

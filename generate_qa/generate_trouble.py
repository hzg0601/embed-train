import __init__
from tools.utils import GLOBAL

import os
import re

from generate_qa.generate_base import GenerateBase
from tools.logger import getLogger

logger = getLogger()

DEFAULT_TROUBLE_GENERATE_PROMPT_TMPL = """
    这是一个施工工地现场的隐患巡检内容，请根据规范原有条文，生成可能的违反该条文的隐患描述，
    隐患可能是安全和质量两个方面，按序号返回每个违背的描述，每条违背内容前加序号，
    每条字数控制在50字以内，一共生成1-3条隐患。
"""


# 隐患数据生成类
class GenerateTrouble(GenerateBase):

    min_dataset_len = 20   # 最小长度，少于这个长度的忽略

    def __init__(self):  # sourcery skip: raise-specific-error
        super().__init__()

    # 生成数据
    def generateData(self, question):

        results = self.generate(question, DEFAULT_TROUBLE_GENERATE_PROMPT_TMPL)

        # 以'\n'为分界，将字符串分割成列表
        results_list = results.split('\n')

        # 创建一个新的列表来存储去除序号的结果
        new_results = []

        try:
            for result in results_list:
                try:
                    if len(result.strip()) < self.min_dataset_len:
                        continue

                    # 使用正则表达式删除 "隐患X：" 和移除开始的数字和点
                    new_result = result.lstrip('1234567890. ')
                    new_result = re.sub(r"隐患\d+：", '', new_result)

                    # 删除常见的前缀并移除空格
                    new_result = new_result.replace("隐患描述：", "").replace("隐患：", "")
                    new_result = new_result.replace("违背描述：", "").replace("违背内容：", "").replace("违背内容描述：", "").strip()

                    new_results.append(new_result)

                    # 输出修改前后内容
                    # if (result != new_result):
                    #     logger.debug(result)
                    #     logger.debug(new_result)
                except Exception as e:
                    logger.error("Error generateData:%s", e)
                    logger.error(result)
                    continue

        except Exception as e:
            logger.error("Error generateData:%s", e)
            logger.error(results)

        return new_results


if __name__ == '__main__':  # sourcery skip: raise-specific-error

    # 提示词
    # question = "6.3.8 挑脚手架房层门架立杆与型钢是挑梁立可接，门架立杆不得滑动或案动，型钢深上应设置定位销位销后与门架立杆的间隙不直大于3mm"

    question = "9.0.10 门式作业脚手架在使用期间，不应拆除加固杆、连墙 件、转角处连接杆、通道口斜撑杆等加固杆件。"

    generate_trouble = GenerateTrouble()
    results = generate_trouble.generateData(question)

    for res in results:
        print(res)
    pass

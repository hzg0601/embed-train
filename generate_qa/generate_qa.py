import __init__
from tools.utils import GLOBAL

import os
import re

from generate_qa.generate_base import GenerateBase
from tools.logger import getLogger

logger = getLogger()


DEFAULT_QA_GENERATE_PROMPT_TMPL = """
    
以下是关键的背景信息。

---------------------
{context_str}
---------------------

根据所提供的背景信息，并在不需要任何外部知识的前提下，
请设想您是一位专注细节的教育者。您的目标是根据背景资料
制定一个准确、具体的问题，用于小型测验或考试环节。
确保这个问题直接关联到文档中明确提到的信息，
并且限制问题必须只针对给出的背景内容。
注意给你的背景信息中，会包含x.x.x这种前缀，请忽略。
并且直接以问题形式呈现答案，避免任何引导性语句或解释性内容。
"""

# 问答数据生成类
class GenerateQA(GenerateBase):

    min_dataset_len = 5   # 最小长度，少于这个长度的忽略

    def __init__(self):  # sourcery skip: raise-specific-error
        super().__init__()

    # 生成数据
    def generateData(self, question):

        results = self.generate(question, DEFAULT_QA_GENERATE_PROMPT_TMPL)

        # 以'\n'为分界，将字符串分割成列表
        results_list = results.split('\n')

        # 创建一个新的列表来存储去除序号的结果
        new_results = []

        try:
            for result in results_list:
                try:
                    if len(result.strip()) < self.min_dataset_len:
                        continue
                    
                    # 删除常见的前缀并移除空格
                    new_result = result.replace("根据所提供的背景信息，以下是制定的问题：","")
                    new_result = new_result.replace("根据提供的信息，","")

                    if len(new_result.strip()) < self.min_dataset_len:
                        continue
                    
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
    # question = "9.0.10 门式作业脚手架在使用期间，不应拆除加固杆、连墙 件、转角处连接杆、通道口斜撑杆等加固杆件。"
    # question = "4.1.1 在建工程不得在外电架空线路正下方施工、搭设作业棚、 建造生活设施或堆放构件、架具、材料及其他杂物等。"
    # question = "1.0.3 建筑施工现场临时用电工程专用的电源中性点直接接地 的 220/380 V 三相四线制低压电力系统，必须符合下列规定： 1 采用三级配电系统； 2 采用TN-S 接零保护系统； 3 采用二级漏电保护系统。"
    question = "3.0.3 机械设备的检查、维修、保养、故障记录，应及时、准 确、完整、字迹清晰。"

    generate_qa = GenerateQA()
    results = generate_qa.generateData(question)

    for res in results:
        print(res)
    pass

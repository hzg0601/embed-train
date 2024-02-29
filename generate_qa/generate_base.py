import __init__
from tools.utils import GLOBAL

import time
from datetime import datetime
import httpx
from openai import OpenAI
import os

open_ai_http_proxy = os.environ.get("open_ai_http_proxy")
open_ai_api_key = os.environ.get("open_ai_api_key")


class GenerateBase():
    client = OpenAI(
        # Or use the `OPENAI_BASE_URL` env var
        base_url="https://api.openai.com/v1",
        http_client=httpx.Client(
            proxies=open_ai_http_proxy,
            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        ),
        api_key=open_ai_api_key,  # this is also the default, it can be omitted
    )

    current_call_count = 1
    repeat_call_count = 5  # 重试次数
    max_question_len = 1000  # 最大问题长度
    call_over_time = 100  # 超时设置，秒

    def inner_generate(self, question, generate_prompt):

        # 获取当前时间
        datetime1 = datetime.now()
        current_time = datetime1.strftime('%H:%M:%S')
        print(f"call gpt begin at {current_time}")

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": generate_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            timeout=self.call_over_time
        )

        # 计算时间差
        datetime2 = datetime.now()
        current_time = datetime2.strftime('%H:%M:%S')
        time_diff = datetime2 - datetime1
        seconds = time_diff.seconds % 60
        print(f"call gpt finished at {current_time},elapsed time(seconds): {seconds}")

        return response.choices[0].message.content

    def inner_generate_repeat(self, question, generate_prompt):
        while (self.current_call_count <= self.repeat_call_count):
            try:
                results = self.inner_generate(question, generate_prompt)

                # 没有异常，表示成功了，直接返回
                return results
            except Exception as e:
                print("Error inner_generate_repeat:", e)

            # 每次重试延迟时间，分别延迟 5,10,15...重试
            time.sleep(5 * self.current_call_count)
            self.current_call_count += 1

        raise Exception(f"Failed to call inner_generate_repeat,question:{question}")

    def smart_truncate(self, content, length=1000, suffix='...'):
        if len(content) <= length:
            return content

        result = ' '.join(content[:length+1].split(' ')[0:-1])

        if len(result) < len(content):
            result += suffix

        return result

    def generate(self, question, generate_prompt) -> []:
        # 调用支持重试的函数，current_call_count需要重新赋值为1，支持重入
        self.current_call_count = 1

        truncated_question = self.smart_truncate(question, self.max_question_len)
        results = self.inner_generate_repeat(truncated_question, generate_prompt)
        return results

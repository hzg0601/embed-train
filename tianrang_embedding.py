import requests
import os
import regex as re
import asyncio
import aiohttp
import pandas as pd 
import uuid
from tqdm import tqdm
from typing import Any, Coroutine, List
import time
import argparse
from llama_index.bridge.pydantic import PrivateAttr,Field
from llama_index.embeddings.base import BaseEmbedding, Embedding
import warnings


warnings.filterwarnings("ignore")
session_timeout = aiohttp.ClientTimeout(total=None)
URL = "https://traimodel.pinming.cn/v1/workflow-messages"
API_KEY_EMBED = "app-PBhu6fYJNiImAj5tU05Vjgyy"

class TianrangLIEmbedding(BaseEmbedding):
    api_key : str = Field(description="Tianrang API KEY")
    url: str = Field(description="Tianrang URL")
    def __init__(self,
                api_key = API_KEY_EMBED,
                url = URL
                ):
        super().__init__(
             api_key=api_key,
             url = url
        )
        self.api_key = api_key
        self.url = url

    def get_prerequisite(self,api_key=API_KEY_EMBED):
        headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        }
        json_data = {
                'inputs': {},
                'query': '',
                'response_mode': 'blocking',
                'user': 'user-001',
                "return chain output": True
                }
        return headers,json_data


    def _query_embedding(self,           
                        query,
                        json_data,
                        headers
                        ):
        session = requests.Session()
        json_data['query'] = query
        response = session.post(
                            url=self.url,
                            headers=headers,
                            json=json_data
                            ).json()
        return response

    async def _aquery_embedding(self,
                                query,
                                json_data,
                                headers
                                ):
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
                    json_data["query"] = query
                    async with session.post(url=self.url,headers=headers,json=json_data) as response:
                        try:
                            return await response.json()
                        except aiohttp.client_exceptions.ContentTypeError:
                            return await response.json(content_type='text/html',encoding='utf-8')
                        except Exception as e:
                            print(e)
                    
    def _get_vector(self,response:dict):
        return eval(response["answer"]["vector"])
    
    def _get_query_embedding(self, query: str) -> List[float]:
        headers, json_data = self.get_prerequisite(self.api_key)
        response = self._query_embedding(query=query,
                                          json_data=json_data,
                                          headers=headers)
        return self._get_vector(response)
    # 用aiohttp与普通http请求每100个请求差别约3.3-4秒，前者约160秒/300请求，后者170秒/300请求
    async def _aget_query_embedding(self, query: str) -> List[float]:
        headers, json_data = self.get_prerequisite(self.api_key)
        response =await asyncio.gather(asyncio.create_task( self._aquery_embedding(query=query,
                                          json_data=json_data,
                                          headers=headers)))
        return self._get_vector(response[0])
        # return self._get_query_embedding(text)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._aget_query_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:

        headers, json_data = self.get_prerequisite(self.api_key)
        tasks = []
        for text in texts:
            tasks.append(asyncio.create_task(self._aquery_embedding(text,json_data,headers)))
        result = await asyncio.gather(*tasks)
        result = [self._get_vector(response) for response in result]
        return result
    

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        result = asyncio.run(self._aget_text_embeddings(texts))
        return result
    
    def _embed(self,texts: List[str]) -> List[List[float]]:
         return self._get_text_embeddings(texts)

if __name__ == "__main__":
    
    embed = TianrangLIEmbedding()
    texts = ["你好",]*300
    result = embed._get_text_embeddings(texts)
    import time 
    st = time.time()
    for text in texts:
        result = asyncio.run(embed._aget_query_embedding(text))
    end = time.time()
    print(f"time elapse:{end-st}")

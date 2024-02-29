import requests
import os
import regex as re
import asyncio
import aiohttp
import pandas as pd 
import uuid
from tqdm import tqdm
from typing import List
import time
import argparse
import warnings
import json

warnings.filterwarnings("ignore")
session_timeout = aiohttp.ClientTimeout(total=None)
parser = argparse.ArgumentParser("batch knowledge engineering")


URL = "https://traimodel.pinming.cn/v1/completion-messages"
API_KEY_QA = 'app-hm6Owh33MSDKjhIsYmykduXa'
API_KEY_SUMMARY = "app-LHF0MqDLRNQIqrNX6vlmU7VL"

# parser.add_argument("--url",type=str,default=URL,help="the target url of traimodel.pingming.cn")
parser.add_argument("--app_type",type=str,default="QA",choices=['SUMMARY','QA'],
                    help="the app type, only support`QA` or `SUMMARY` for now ")
parser.add_argument("--keep_items",type=int,default=1,
                    help="the number of query to keep ")

API_KEY_DICT = {"QA":API_KEY_QA, "SUMMARY":API_KEY_SUMMARY}
COL_NAME_DICT = {"QA":"Q","SUMMARY":"摘要"}
# FILE_KEY_DICT = {""}
PROJECT_PATH = os.path.dirname(__file__)
IN_DATA_DIR = os.path.join(PROJECT_PATH,"data/raw_knowledge/")
OUT_DATA_DIR = os.path.join(PROJECT_PATH,"data/knowledge_qa/")
if not os.path.exists(IN_DATA_DIR):
    os.makedirs(IN_DATA_DIR)
if not os.path.exists(OUT_DATA_DIR):
    os.makedirs(OUT_DATA_DIR)
class KnowledgeEngineering(object):
    def __init__(self,
                 in_data_dir=IN_DATA_DIR,
                #  file_name_reg=".*",
                 out_data_dir=OUT_DATA_DIR,
                #  api_key=API_KEY_QA,
                keep_items=1,
                 url=URL) -> None:
        self.in_data_dir = in_data_dir
        # self.file_name_reg = file_name_reg
        self.out_data_dir = out_data_dir
        # self.api_key=api_key
        self.url=url
        self.keep_items = keep_items

    def file_reader(self,data_dir=IN_DATA_DIR,file_name_reg=".*")->dict:

        file_list = [file for file in 
            os.listdir(data_dir) if re.search(file_name_reg, file)]
        data_df = {}
        for file in file_list:
            full_file_path = os.path.join(data_dir,file)
            raw_file = pd.ExcelFile(full_file_path)
            sheets = raw_file.sheet_names
            temp_df = []
            for sheet in sheets:
                temp = pd.read_excel(full_file_path,sheet_name=sheet)
                temp.rename(columns={"切片内容":"分片内容"},inplace=True)
                temp = temp.dropna(subset=["分片内容"])
                temp_df.append(temp)
            # 关闭，否则会报出too many files
            raw_file.close()
            temp_df = pd.concat(temp_df,ignore_index=True,axis=0)
            data_df[file] = temp_df

        return data_df

    def get_prerequisite(self,        
            api_key=API_KEY_QA,
            in_data_dir=IN_DATA_DIR,
            file_name_reg=".*"
                        ):
        data_df = self.file_reader(data_dir=in_data_dir,file_name_reg=file_name_reg)

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
        return data_df,headers,json_data

    def process_response(self,response, chunk:str,doc_id:str,col_name:str="Q",keep_items=1):
        response = response["answer"]
        if col_name == "Q":
            questions = re.sub("Q[1-9]: ","",response).split("\n\n")[:keep_items]
        elif col_name == "摘要":
             questions = [response]
        length = len(questions)
        corpus = [chunk]*length
        corpus_ids = [str(uuid.uuid4())]*length
        doc_ids = [doc_id]*length
        temp = {"文档id":doc_ids,
                "切片id":corpus_ids,
                "切片内容":corpus,
                col_name:questions,
                }
        temp = pd.DataFrame(temp)
        return temp

    def postprocess_one_file(self,
                             result:List[pd.DataFrame],
                             out_data_dir:str,
                             file_name:str,
                             file_name_key:str="QA"
                             ):
            result = pd.concat(result,axis=0,ignore_index=True)
            file_name = re.sub("分片|分片内容|切片|切片内容",file_name_key,file_name)
            out_path = os.path.join(out_data_dir,file_name)
            result.to_excel(out_path,index=True)
            print(f"file {file_name} done.")


    def batch(self,
            url=URL,
            api_key=API_KEY_QA,
            in_data_dir=IN_DATA_DIR,
            file_name_reg=".*",
            out_data_dir=OUT_DATA_DIR,
            col_name:str="Q",
            file_name_key:str="QA",
            keep_items:int=1
            ):
        print("start to do batch_knowledge_engineering...")
        time_st = time.time()
        data_df,headers,json_data = self.get_prerequisite(
                                                    api_key=api_key,
                                                    in_data_dir=in_data_dir,
                                                    file_name_reg=file_name_reg
                                                    )
        session = requests.Session()
        
        for file_name, doc in data_df.items():
            print(f"start to process file：{file_name}")
            result = []
            doc_id = str(uuid.uuid4())
            for chunk in tqdm(doc["分片内容"]):
                json_data["query"] = chunk
                try:
                    response = session.post(
                                            url=url,
                                            headers=headers,
                                            json=json_data
                                            ).json()
                    temp = self.process_response(response=response,
                                                 chunk=chunk,
                                                 doc_id=doc_id,
                                                 col_name=col_name,
                                                 keep_items=keep_items
                                                 )
                    result.append(temp)
                except aiohttp.client_exceptions.ContentTypeError as e:
                    print("ERROR OCCURED!!!")
                    print(e)
                    response = session.post(
                                            url=url,
                                            headers=headers,
                                            json=json_data
                                            ).json(content_type='text/html',encoding='utf-8')
                    temp = self.process_response(response=response,
                                                 chunk=chunk,
                                                 doc_id=doc_id,
                                                 col_name=col_name,
                                                 keep_items=keep_items
                                                 )
                    result.append(temp)                    
                except Exception as e:
                    print(e)
            self.postprocess_one_file(result=result,
                                      out_data_dir=out_data_dir,
                                      file_name=file_name,
                                      file_name_key=file_name_key)
        time_end = time.time()
        print(f"time elapse: {time_end-time_st} s")
        print("all file processed.")  

    async def post_query_init_session(self,
                         chunk,
                         json_data,
                         url,
                         headers
                         ):
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
                    json_data["query"] = chunk
                    async with session.post(
                                            url=url,
                                            headers=headers,
                                            json=json_data
                                                ) as response:
                        try:
                            return await response.json()
                        except aiohttp.client_exceptions.ContentTypeError as e:
                            print("WARNING.....")

                            return await response.json(content_type='text/html') #,encoding='utf-8'                
                        except json.decoder.JSONDecodeError as e:
                            print("ERROR...")

                        except Exception as e:
                            print(e)
    async def post_query(self,
                         chunk,
                         json_data,
                         url,
                         headers,
                         session
                         ):

                json_data["query"] = chunk
                try_iterations = 0
                while try_iterations <= 4:
                    try_iterations += 1
                    response = await session.post(
                    url=url,
                    headers=headers,
                    json=json_data
                        )
                    try:
                        return await response.json(content_type='application/json')
                    except aiohttp.client_exceptions.ContentTypeError:
                        print("application/json ContentTypeError WARNING, repost.....")
                        text = await response.text()
                        print(f"THE RETURN TEXT IS:{text}")
                response.close()

    async def abatch(self,
            url=URL,
            api_key=API_KEY_QA,
            in_data_dir=IN_DATA_DIR,
            file_name_reg=".*",
            out_data_dir=OUT_DATA_DIR,
            file_name_key:str="QA",
            col_name:str="Q",
            keep_items:int=1
                     ):
        print("start to do batch_knowledge_engineering...")
        time_st = time.time()
        data_df,headers,json_data = self.get_prerequisite(
                                                    api_key=api_key,
                                                    in_data_dir=in_data_dir,
                                                    file_name_reg=file_name_reg
                                                    )
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            for file_name, doc in data_df.items():
                print(f"start to process file：{file_name}")
                time_st_temp = time.time()
                doc_id = str(uuid.uuid4())
                tasks = []
                for chunk in tqdm(doc["分片内容"]):
                    tasks.append(asyncio.create_task(self.post_query(chunk,json_data,url,headers,session)))
                result = await asyncio.gather(*tasks)
                result = [self.process_response(response,chunk,doc_id,col_name=col_name,keep_items=keep_items) 
                        for response,chunk in zip(result,doc["分片内容"])]
                self.postprocess_one_file(result=result,
                                            out_data_dir=out_data_dir,
                                            file_name=file_name,
                                            file_name_key=file_name_key)
                time_end_temp = time.time()
                print(f"file {file_name} time elapse: {time_end_temp-time_st_temp} s")
        time_end = time.time()
        print(f"all files time elapse: {time_end-time_st} s")
        print("all file processed.")  

          
if __name__ == "__main__":
    args = parser.parse_args()
    ke = KnowledgeEngineering()
    # ke.batch(
    #          api_key=API_KEY_DICT[args.app_type],
    #          col_name=COL_NAME_DICT[args.app_type],
    #          file_name_key=args.app_type
    #          )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(ke.abatch(
             api_key=API_KEY_DICT[args.app_type],
             col_name=COL_NAME_DICT[args.app_type],
             file_name_key=args.app_type,
             keep_items=args.keep_items
    ))
    loop.close()


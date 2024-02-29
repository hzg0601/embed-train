# 1.数据集说明和下载

| 数据集名称             | 下载地址                                                                                                                                                                                  | 数据集说明                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| datasets_qa.csv        | [下载](http://pmbimcloud-ai.oss-cn-hangzhou.aliyuncs.com/nlp%2Fdatasets%2Fdatasets_qa.csv?OSSAccessKeyId=LTAI8PPrIEVEAQCK&Expires=1706554903&Signature=%2F7sdNwF7BLiG0eRRsRkJKnQ3DlQ%3D)     | gpt3.5生成，38本巡检宝规范的问答对，约19000条数据集                    |
| datasets_qa.jsonl      | [下载](http://pmbimcloud-ai.oss-cn-hangzhou.aliyuncs.com/nlp%2Fdatasets%2Fdatasets_qa.jsonl?OSSAccessKeyId=LTAI8PPrIEVEAQCK&Expires=1703314608&Signature=POCsZQdx3fqUJmIn3pYWS9daXCI%3D)     | 同上，jsonl格式                                                        |
| datasets_qa_v1.0.csv   | 略                                                                                                                                                                                        | gpt3.5生成，38本巡检宝规范的问答对（删减），约15000条数据集            |
| datasets_qa_v1.0.jsonl | 略                                                                                                                                                                                        | 同上，jsonl格式                                                        |
| datasets_v1.0.csv      | [下载](http://pmbimcloud-ai.oss-cn-hangzhou.aliyuncs.com/nlp%2Fdatasets%2Fdatasets_v1.0.csv?OSSAccessKeyId=LTAI8PPrIEVEAQCK&Expires=1703314705&Signature=vnH5rQ5rQDNPS5YR%2FVZUyN0r%2FTI%3D) | gpt3.5生成，38本巡检宝规范的隐患问答对，约25000条数据集                |
| datasets_v1.0.jsonl    | [下载](http://pmbimcloud-ai.oss-cn-hangzhou.aliyuncs.com/nlp%2Fdatasets%2Fdatasets_v1.0.jsonl?OSSAccessKeyId=LTAI8PPrIEVEAQCK&Expires=1703314726&Signature=pnOtCl6W8ldBswwAPhZd%2BcXzmZU%3D) | 同上，jsonl格式                                                        |
| datasets_v1.1.csv      | [下载](http://pmbimcloud-ai.oss-cn-hangzhou.aliyuncs.com/nlp%2Fdatasets%2Fdatasets_v1.1.csv?OSSAccessKeyId=LTAI8PPrIEVEAQCK&Expires=1706554794&Signature=R4fr3o0gC3aCBkLdd6OwmDUsPJg%3D)     | datasets_v1.0.csv+28条 《AI眼镜规范查询测试汇总.xlsx》中的数据集       |
| datasets_v1.1.jsonl    | [下载](http://pmbimcloud-ai.oss-cn-hangzhou.aliyuncs.com/nlp%2Fdatasets%2Fdatasets_v1.1.jsonl?OSSAccessKeyId=LTAI8PPrIEVEAQCK&Expires=1706554816&Signature=UlULotsHKw1Cr1XEyzwPWzmdciI%3D)   | 同上，jsonl格式                                                        |
| datasets_v2.0.csv      | 略                                                                                                                                                                                        | 算法组手动标注隐患数据（暂不提供下载）                                 |
| datasets_v2.0.jsonl    | 略                                                                                                                                                                                        | 同上，jsonl格式                                                        |
| datasets_v2.1.csv      | 略                                                                                                                                                                                        | 算法组手动标注隐患数据 + 28条《AI眼镜规范查询测试汇总.xlsx》中的数据集 |
| datasets_v2.1.jsonl    | 略                                                                                                                                                                                        | 同上，jsonl格式                                                        |

# 2.训练方法说明

目标：

1.支持微调m3e-base和bge-large嵌入模型，优先保证m3e-base模型的微调

2.支持数据集的生成和管理

3.对微调后的模型进行自动评估（基于巡检数据库）

## 2.1 baai_train.py

以北京智能研究院的FlagEmbedding方式微调，也支持bge和m3e嵌入模型的训练，当前没有整合批量训练方式

## 2.2 uniem_train.py

以M3E嵌入模型官方推荐的uniem方式微调。

* 支持bge和m3e嵌入模型的训练
* 支持数据集合并
* 支持csv和jsonl格式
* 支持批量训练

说明：微调m3e嵌入模型，效果还可以，其中sv格式训练最为推荐，结果好于jsonl格式。

## 2.3 llama_index_train.py

基于llama-index框架提供的训练方法，使用的是特有的数据格式，训练流程没有完全走通，问题在于微调后的权重文件是单独的，出在合并成统一的权重文件，当前暂缓，后续有精力再探索。

# 3.自动评估数据集

evaluate/test_asset文件夹下，当前有两个文件，分别如下：

AI眼镜规范查询测试汇总.xlsx：28条隐患检索规范及参考答案（数据集太少，作为评估不充分）

AI眼镜规范查询测试汇总_2.0.xlsx：28条隐患检索规范及参考答案 + 基于2000条标注的隐患数据调整

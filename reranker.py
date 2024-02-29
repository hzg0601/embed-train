from langchain.retrievers.document_compressors import CohereRerank

from llama_index.postprocessor import SentenceTransformerRerank
from sentence_transformers import SentenceTransformer,CrossEncoder
from typing import Optional, Sequence

from langchain_core.documents import Document

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
import os
model_path = os.environ["HOME"]+"/models/bge-reranker-large/"
# instruction = "为这个句子生成表示以用于检索相关文章："


# reranker = SentenceTransformerRerank(
#     top_n=5,
#     model=model_path,
# )
# reranker._model.predict([["我是","你是"],["我就是","他是"]],convert_to_tensor=True)

# print("Load reranker")



class LangchainRerank(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""

    def __init__(self,
                 top_n:int=3, 
                 device:str="cuda", 
                 max_length:int=1024,               
                 batch_size: int = 32,
                show_progress_bar: bool = None,
                num_workers: int = 0,
                activation_fct = None,
                apply_softmax = False,
                model_name_or_path:str="BAAI/bge-reraker-large"):
        self.top_n=top_n
        self.model_name_or_path=model_name_or_path

        self.device=device
        self.max_length=max_length
        self.batch_size=batch_size
        self.show_progress_bar=show_progress_bar
        self.num_workers=num_workers
        self.activation_fct=activation_fct
        self.apply_softmax=apply_softmax

        self.model = CrossEncoder(model_name=model_path,max_length=1024,device=device)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        sentence_pairs = [[query,_doc] for _doc in _docs]
        results = self.model.predict(sentences=sentence_pairs,
                                     batch_size=self.batch_size,
                                     show_progress_bar=self.show_progress_bar,
                                     num_workers=self.num_workers,
                                     activation_fct=self.activation_fct,
                                     apply_softmax=self.apply_softmax,
                                     convert_to_tensor=True)
        top_k = self.top_n if self.top_n < len(results) else len(results)
        
        values, indices = results.topk(top_k)
        final_results = []
        for value, index in zip(values,indices):
            doc = doc_list[index]
            doc.metadata["relevance_score"] = value
            final_results.append(doc)
        return final_results
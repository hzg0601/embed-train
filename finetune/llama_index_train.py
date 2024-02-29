import __init__
import os
import shutil
from tools.utils import GLOBAL

import torch

from llama_index.finetuning import EmbeddingQAFinetuneDataset

from llama_index.finetuning import EmbeddingAdapterFinetuneEngine, SentenceTransformersFinetuneEngine
from llama_index.embeddings import resolve_embed_model, AdapterEmbeddingModel
from typing import cast
from sentence_transformers import SentenceTransformer


class FineTuneModel:

    base_model_path = None
    finetune_model_path = None
    adapter_finetune_model_path = None

    def __init__(self):
        super().__init__()

    def copy_model(self):
        # 确保目标目录存在，如果不存在则创建它
        if not os.path.exists(self.finetune_model_path):
            os.makedirs(self.finetune_model_path)

        # 遍历源目录中的所有文件
        for filename in os.listdir(self.base_model_path):
            # model.safetensors不会更改，且会被优先加载，所以忽略拷贝
            if filename == "model.safetensors":
                continue

            src_file = os.path.join(self.base_model_path, filename)

            # 如果是文件，则复制到目标目录
            if os.path.isfile(src_file):
                dst_file = os.path.join(self.finetune_model_path, filename)
                shutil.copy2(src_file, dst_file)  # copy2保留元数据

    def copy_config_file(self):
        filename = "config.json"
        src_file = os.path.join(self.base_model_path, filename)

        # 如果是文件，则复制到目标目录
        if os.path.isfile(src_file):
            dst_file = os.path.join(self.adapter_finetune_model_path, filename)
            shutil.copy2(src_file, dst_file)  # copy2保留元数据

    def delete_unused_model(self):
        filename = "model.safetensors"
        model_file = os.path.join(self.finetune_model_path, filename)

        if os.path.isfile(model_file) and os.path.exists(model_file):
            os.remove(model_file)

    def train_model_adapter(self):
        train_dataset_file = os.path.join(GLOBAL.DATA_PATH, "datasets_v1.1.json")
        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_file)

        base_embed_model = resolve_embed_model(f"local:{self.base_model_path}")
        embed_model_output_path: str = self.adapter_finetune_model_path
        if not os.path.exists(embed_model_output_path):
            os.mkdir(embed_model_output_path)

        finetune_engine = EmbeddingAdapterFinetuneEngine(
            train_dataset,
            base_embed_model,
            model_output_path=embed_model_output_path,
            # bias=True,
            epochs=5,
            verbose=True,
            # optimizer_class=torch.optim.SGD,
            # optimizer_params={"lr": 0.01}
        )

        finetune_engine.finetune()

        # config.json会有错误，用原来的覆盖
        self.copy_config_file()

        # model.safetensors会被优先加载，训练后保证删除  备注：前面复制模型的时候已经不拷贝了，理论上不会出现存在这个文件的情况
        self.delete_unused_model()
        
        # 合并模型
        self.merge_model()

    def train_model_sentence_transformers(self):
        train_dataset_file = os.path.join(GLOBAL.DATA_PATH, "datasets_v1.0.json")
        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_file)

        embed_model_output_path: str = self.adapter_finetune_model_path
        if not os.path.exists(embed_model_output_path):
            os.mkdir(embed_model_output_path)

        train_dataset_file = os.path.join(GLOBAL.DATA_PATH, "datasets_v1.0_eval.json")
        val_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_file)
        finetune_engine = SentenceTransformersFinetuneEngine(
            train_dataset,
            model_id=self.base_model_path,
            model_output_path=embed_model_output_path,
            val_dataset=val_dataset,
        )

        finetune_engine.finetune()
        
        # 这里是基础模型，只有修改的部分  TODO  这里汇总保存的应该还是有问题
        embed_model = finetune_engine.get_finetuned_model()
        embedder = cast(SentenceTransformer, embed_model)
        embedder.save(str(self.finetune_model_path))

    # def merge_model_deprecated():
    #     import torch
    #     from transformers import AutoModel

    #     # 加载基础模型
    #     base_model = AutoModel.from_pretrained(self.base_model_path)

    #     # 假设 lora_state_dict 包含了LORA适配器的权重
    #     lora_file_name = os.path.join(self.adapter_finetune_model_path, "pytorch_model.bin")
    #     lora_state_dict = torch.load(lora_file_name)

    #     # 更新基础模型的权重
    #     base_model_state_dict = base_model.state_dict()
    #     base_model_state_dict.update(lora_state_dict)  # 这将覆盖掉原有的权重
    #     missing_keys, unexpected_keys = base_model.load_state_dict(base_model_state_dict, strict=False)

    #     # 检查是否有缺失或未预期的键
    #     print("Missing keys:", missing_keys)
    #     print("Unexpected keys:", unexpected_keys)

    #     # 保存整合了LORA权重的新模型
    #     saved_file_name = os.path.join(self.finetune_model_path, "pytorch_model_temp.bin")
    #     torch.save(base_model_state_dict, saved_file_name)

    def merge_model(self):
        # 先复制所有原始的模型权重文件，后续进行模型合并
        self.copy_model()

        # 加载基础模型的权重（例如，预训练模型）
        base_file_name = os.path.join(self.base_model_path, "pytorch_model.bin")
        base_model_state_dict = torch.load(base_file_name)

        # 加载Adapter模型的权重（已经过微调）
        adapter_file_name = os.path.join(self.adapter_finetune_model_path, "pytorch_model.bin")
        adapter_model_state_dict = torch.load(adapter_file_name)

        # 创建新的状态字典
        merged_state_dict = base_model_state_dict.copy()

        # 更新状态字典以包含Adapter的权重
        for key, value in adapter_model_state_dict.items():
            # if "adapter" in key:  # 假设Adapter层的参数键值名称中含有'adapter'  备注：微调后，只有一个key，名称为：linear.weight  TODO  待确认微调逻辑是否正确
            merged_state_dict[key] = value

        # 保存合并后的模型权重
        saved_file_name = os.path.join(self.finetune_model_path, "pytorch_model.bin")
        torch.save(merged_state_dict, saved_file_name)

    # def merge_model_by_cocktail(self):
    #     from LM_Cocktail import mix_models, mix_models_with_data

    #     # Mix fine-tuned model and base model; then save it to output_path: ./mixed_model_1
    #     model = mix_models(
    #         model_names_or_paths=[self.base_model_path, self.adapter_finetune_model_path],
    #         model_type='encoder',
    #         weights=[0.5, 0.5],  # you can change the weights to get a better trade-off.
    #         output_path=self.finetune_model_path)


class FineTuneM3EModel(FineTuneModel):

    def __init__(self):
        self.base_model_path = GLOBAL.M3E_BASE_MODEL_PATH
        self.finetune_model_path = GLOBAL.M3E_FINETUNE_MODEL_PATH
        self.adapter_finetune_model_path = GLOBAL.M3E_ADAPTER_FINETUNE_MODEL_PATH
        super().__init__()


class FineTuneBGEModel(FineTuneModel):

    def __init__(self):
        self.base_model_path = GLOBAL.BGE_BASE_MODEL_PATH
        self.finetune_model_path = GLOBAL.BGE_FINETUNE_MODEL_PATH
        self.adapter_finetune_model_path = GLOBAL.BGE_ADAPTER_FINETUNE_MODEL_PATH

        super().__init__()


def train_m3e_model_adapter():
    model = FineTuneM3EModel()
    model.train_model_adapter()
    
def train_m3e_model_sentence_transformers():
    model = FineTuneM3EModel()
    model.train_model_sentence_transformers()


def train_bge_model():
    model = FineTuneBGEModel()
    model.train_model_adapter()


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    # train_m3e_model_adapter()
    train_m3e_model_sentence_transformers()
    
    # train_bge_model()
    pass

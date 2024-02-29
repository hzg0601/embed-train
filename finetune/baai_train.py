import __init__ # noqa
import os  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # noqa

from tools.utils import GLOBAL
import subprocess

from evaluate.evaluate_model import EvaluateModel, EvaluateMethodType

from tools.logger import getLogger


logger = getLogger()

class FineTuneModel(EvaluateModel):

    base_model_path = None
    finetune_model_path = None

    JSONL_DATA_SETS_NAME = 'datasets_v2.2.jsonl'
    learning_rate = "1e-5"
    num_train_epochs = "5"
    per_device_train_batch_size = "5"
    query_instruction_for_retrieval = "为这个句子生成表示以用于检索相关文章："
    evaluate_method_type: EvaluateMethodType = EvaluateMethodType.llama_index
    json_file_name = 'datasets_v2.2.json'

    def __init__(self):
        super().__init__()

    def train_model(self):
        # 参考链接：https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
        # per_device_train_batch_size: batch size in training. In most of cases, larger batch size will bring stronger performance. You can expand it by
        #                             enabling --fp16, --deepspeed ./df_config.json (df_config.json can refer to ds_config.json), --gradient_checkpointing, etc.
        # train_group_size: the number of positive and negatives for a query in training. There are always one positive, so this argument will control the number of negatives
        #                             (#negatives=train_group_size-1). Noted that the number of negatives should not be larger than the numbers of negatives in data "neg":List[str].
        #                             Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
        # learning_rate: select a appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale.
        # query_max_len: max length for query. Please set it according the average length of queries in your data.
        # passage_max_len: max length for passage. Please set it according the average length of passages in your data.

        # Parameters for training
        model_name = self.base_model_path
        output_dir = self.finetune_model_path
        train_data = os.path.join(GLOBAL.DATA_PATH, self.JSONL_DATA_SETS_NAME)

        fp16 = True

        dataloader_drop_last = True
        normalized = True
        temperature = "0.02"
        query_max_len = "256"
        passage_max_len = "512"
        train_group_size = "2"
        negatives_cross_device = True
        
        save_steps = "1000000"
        logging_steps = "1000"
        
        # 使用GPU的数量
        num_gpus = "1"

        # Constructing the command
        command = [
            "torchrun", "--nproc_per_node", num_gpus,
            "-m", "FlagEmbedding.baai_general_embedding.finetune.run",
            "--output_dir", output_dir,
            "--model_name_or_path", model_name,
            "--save_steps", save_steps,
            "--train_data", train_data,
            "--learning_rate", self.learning_rate,
            "--fp16" if fp16 else "",
            "--num_train_epochs", self.num_train_epochs,
            "--per_device_train_batch_size", self.per_device_train_batch_size,
            "--dataloader_drop_last" if dataloader_drop_last else "",
            "--normlized" if normalized else "",
            "--temperature", temperature,
            "--query_max_len", query_max_len,
            "--passage_max_len", passage_max_len,
            "--train_group_size", train_group_size,
            "--negatives_cross_device" if negatives_cross_device else "",
            "--logging_steps", logging_steps,
            "--query_instruction_for_retrieval", self.query_instruction_for_retrieval,
        ]

        # Filter out any empty strings from the command list
        command = [arg for arg in command if arg]

        # Execute the command
        subprocess.run(command, check=True)

        # Eval script (commented as it was in the original shell script)
        # subprocess.run(["python", "eval_llama_index.py"], check=True)

    def evaluate_model(self):
        pass


class FineTuneM3EModel(FineTuneModel):

    def __init__(self):
        self.base_model_path = GLOBAL.M3E_BASE_MODEL_PATH
        self.finetune_model_path = GLOBAL.M3E_FINETUNE_MODEL_PATH

        self.learning_rate = "3e-5"
        self.num_train_epochs = "30"
        self.per_device_train_batch_size = "16"
        self.query_instruction_for_retrieval = "''"
        super().__init__()

    def evaluate_model(self):
        train_params = f"method:flagembedding,model:m3e-base,datasets:{self.JSONL_DATA_SETS_NAME},epochs={self.num_train_epochs},batch_size={self.per_device_train_batch_size},lr={self.learning_rate}"
        self.evaluate_m3e(train_params, self.finetune_model_path, self.json_file_name, self.evaluate_method_type)


class FineTuneBGEModel(FineTuneModel):

    def __init__(self):
        self.base_model_path = GLOBAL.BGE_BASE_MODEL_PATH
        self.finetune_model_path = GLOBAL.BGE_FINETUNE_MODEL_PATH
        
        self.learning_rate = "1e-5"
        self.num_train_epochs = "30"
        self.per_device_train_batch_size = "5"

        super().__init__()

    def evaluate_model(self):
        train_params = f"method:flagembedding,model:bge-large,datasets:{self.JSONL_DATA_SETS_NAME},epochs={self.num_train_epochs},batch_size={self.per_device_train_batch_size},lr={self.learning_rate}"
        self.evaluate_bge(train_params, self.finetune_model_path, self.json_file_name, self.evaluate_method_type)


def train_m3e_model():
    model = FineTuneM3EModel()
    model.train_model()
    model.evaluate_model()


def train_bge_model():
    model = FineTuneBGEModel()
    model.train_model()
    model.evaluate_model()


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    # train_m3e_model()
    train_bge_model()
    pass

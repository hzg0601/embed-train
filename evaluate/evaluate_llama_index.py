import __init__
from tools.utils import GLOBAL
from eval_llama_index import li_eval_finetune


class EvaluateLlamaIndex():
    @classmethod
    def eval_model(self, train_params, model_base_name, finetune_model_path, json_file_name):
        li_eval_finetune(finetune_class="full", adapter_class="Linear", model_name=model_base_name,
                         model_output_path=finetune_model_path, file_name_reg=json_file_name, disable_reranker=True,
                         train_params=train_params, eval_ks=[1, 5, 10])

    @classmethod
    def evaluate_m3e(self, train_params, finetune_model_path, json_file_name):
        model_base_name = f"{GLOBAL.M3E_BASE_MODEL_PATH}"
        self.eval_model(train_params, model_base_name, finetune_model_path, json_file_name)

    @classmethod
    def evaluate_bge(self, train_params, finetune_model_path, json_file_name):
        model_base_name = f"{GLOBAL.BGE_BASE_MODEL_PATH}"
        self.eval_model(train_params, model_base_name, finetune_model_path, json_file_name)


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    EvaluateLlamaIndex.evaluate_m3e()

    # EvaluateLlamaIndex.evaluate_bge()

    pass

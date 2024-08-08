import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch
from transformers.utils import is_flash_attn_2_available

print(torch.cuda.is_available())
print(torch.cuda.device_count())

class Gemma2B(LLM):

    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    pipeline: pipeline = None
    history : list = None

    def __init__(self, model_name_or_path: str):
        super().__init__()
        print("Download from local...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipeline = pipeline("text-generation", model=model_name_or_path,
                                 model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.history = []
        print("Complete")

    def build_input(self, prompt, role="user"):

        prompt = {"role": role, "content": prompt}
        self.history.append(prompt)
        if role != "user":
            return None
        prompt = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        # print(prompt)
        # inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        # print(inputs)
        return prompt

    def clear_history(self):
        """
        Clears the history used for processing prompts.
        """
        self.history = []

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        messages = self.build_input(prompt, "user")
        prompt = messages
        # prompt : str
        outputs = self.pipeline(
            prompt,
            max_new_tokens=512,
            add_special_tokens=True,
            do_sample=True,
            temperature=0.0001,
            top_k=50,
            top_p=0.95
        )
        response = outputs[0]["generated_text"][len(prompt):]
        self.build_input(response, "assistant")
        return response

    @property
    def _llm_type(self) -> str:
        return "Gemma2B"


class Qwen2_LLM(LLM):

    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    history: List = []

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        print("Download from local...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path,
                                                          device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        self.history = []
        print("Complete")

    def clear_history(self):
        """
        Clears the history used for processing prompts.
        """
        self.history = []

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        self.history.append({"role": "user", "content": prompt})
        print("history")
        # print(self.history)
        input_ids = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True,
                                            top_p=0.9, temperature=0.0001, repetition_penalty=1.1)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # print(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.history.append({"role": "assistant", "content": response})
        return response

    @property
    def _llm_type(self) -> str:
        return "Qwen2_LLM"

class Gemma2_LLM(LLM):

    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    history: List = []

    def __init__(self, mode_name_or_path: str):
        super().__init__()


        print("Create tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path)
        self.history = []
        print("Create model...")
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, device_map="auto")
        # self.device = torch.device("cuda:1")
        # self.model = self.model.to(self.device)
    def clear_history(self):
        """
        Clears the history used for processing prompts.
        """
        self.history = []

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        self.history.append({"role": "user", "content": prompt})
        # print(self.history)
        input_ids = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True,
                                            top_p=0.9, temperature=0.0001, repetition_penalty=1.1)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # print(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.history.append({"role": "assistant", "content": response})
        return response

    @property
    def _llm_type(self) -> str:
        return "Gemma2_LLM"


class ChatGLM4_LLM(LLM):

    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    gen_kwargs: dict = None
    history: List = []

    def __init__(self, mode_name_or_path: str, gen_kwargs: dict = None):
        super().__init__()
        print("Download from local...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        print("Complete")
        self.history = []

        if gen_kwargs is None:
            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        self.gen_kwargs = gen_kwargs

    def clear_history(self):
        """
        Clears the history used for processing prompts.
        """
        self.history = []

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        messages = {"role": "user", "content": prompt}
        self.history.append(messages)
        model_inputs = self.tokenizer.apply_chat_template(
            self.history, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).to('cuda')

        generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.history.append({"role": "assistant", "content": response})
        return response

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": "glm-4-9b-chat",
            "max_length": self.gen_kwargs.get("max_length"),
            "do_sample": self.gen_kwargs.get("do_sample"),
            "top_k": self.gen_kwargs.get("top_k"),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"

if __name__ == '__main__':
    print(is_flash_attn_2_available())
    model_path = ""
    llm = ChatGLM4_LLM(model_path)
    for name, param in llm.model.named_parameters():
        print(f"{name} is on {param.device}")
    # test(chatGLM4_path)
    a = "1 + 1 = ?"
    print(f"{llm.invoke(a)}")
    print("--------------------------")
    a = "What day is it today?"
    print(f"{llm.invoke(a)}")
    print("--------------------------")
    # a = "Are you hungry?"
    # print(f"{llm.invoke(a)}")
    # print("--------------------------")
    # a = "Repeat your response to the first question"
    # print(f"{llm.invoke(a)}")
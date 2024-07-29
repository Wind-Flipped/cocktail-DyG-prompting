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



from concurrent.futures import ThreadPoolExecutor
#


class LLaMA3_LLM(LLM):
    # 基于本地 llama3 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    history: List = []

    def __init__(self, mode_name_or_path: str):

        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path,
                                                          device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.history = []

        print("完成本地模型的加载")

    def build_input(self, prompt):
        user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
        assistant_format = '<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>'
        self.history.append({'role': 'user', 'content': prompt})
        prompt_str = ''
        # 拼接历史对话
        for item in self.history:
            if item['role'] == 'user':
                prompt_str += user_format.format(content=item['content'])
            else:
                prompt_str += assistant_format.format(content=item['content'])
        return prompt_str

    def clear_history(self):
        """
        Clears the history used for processing prompts.
        """
        self.history = []

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):

        input_str = self.build_input(prompt=prompt)
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False, max_length=4096, truncation=True,
                                          return_tensors='pt').to(
            self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=True,
            top_p=0.9, temperature=0.0001, repetition_penalty=1.1, eos_token_id=self.tokenizer.encode('<|eot_id|>')[0]
        )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = self.tokenizer.decode(outputs).strip().replace('<|eot_id|>', "").replace(
            '<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()
        self.history.append({'role': 'assistant', 'content': response})
        return response

    @property
    def _llm_type(self) -> str:
        return "LLaMA3_LLM"

'''
class Gemma2B(LLM):
    # 基于本地 gemma2b 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    pipeline: pipeline = None
    history : list = None

    def __init__(self, model_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipeline = pipeline("text-generation", model=model_name_or_path,
                                 model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.history = []
        print("完成本地模型的加载")

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

'''

class Qwen2_LLM(LLM):
    # 基于本地 Qwen2 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    history: List = []

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path,
                                                          device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        self.history = []
        print("完成本地模型的加载")

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
    # 基于本地 Gemma2 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    history: List = []

    def __init__(self, mode_name_or_path: str):
        super().__init__()

        # 加载预训练的分词器和模型
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

class Phi3Small_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    history: List = []

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, device_map="auto", trust_remote_code=True, attn_implementation="flash_attention_2")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model = self.model.eval()
        self.history = []
        print("完成本地模型的加载")

    def clear_history(self):
        """
        Clears the history used for processing prompts.
        """
        self.history = []

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        self.history.append({"role": "user", "content": prompt})
        # 调用模型进行对话生成
        input_ids = self.tokenizer.apply_chat_template(conversation=self.history, tokenize=True,
                                                       add_generation_prompt=True,
                                                       return_tensors='pt')
        output_ids = self.model.generate(input_ids.to('cuda'), max_new_tokens=512, do_sample=True,
                                         top_p=0.9, temperature=0.0001, repetition_penalty=1.1)

        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": response})
        return response

    @property
    def _llm_type(self) -> str:
        return "Phi3Small_LLM"


class ChatGLM4_LLM(LLM):
    # 基于本地 ChatGLM4 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    gen_kwargs: dict = None
    history: List = []

    def __init__(self, mode_name_or_path: str, gen_kwargs: dict = None):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        print("完成本地模型的加载")
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
        """返回用于识别LLM的字典,这对于缓存和跟踪目的至关重要。"""
        return {
            "model_name": "glm-4-9b-chat",
            "max_length": self.gen_kwargs.get("max_length"),
            "do_sample": self.gen_kwargs.get("do_sample"),
            "top_k": self.gen_kwargs.get("top_k"),
        }

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"

class Vicuna_LLM(LLM):
    # 基于本地 Qwen2 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    history: List = []

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path,
                                                          device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        self.history = []
        print("完成本地模型的加载")

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



def test(model_path):
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    query = "How are you ?"

    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )

    inputs = inputs.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    print(is_flash_attn_2_available())
    llama3_path = "/home/work/LargeModels/LLaMA3/LLM-Research/Meta-Llama-3-8B-Instruct"
    phi3_path = "/home/work/chenyo/myPaper/model/LLM-Research/Phi-3-small-8k-instruct"
    gemma2_path = "/home/work/chenyo/myPaper/model/LLM-Research/gemma-2-9b-it"
    chatGLM4_path = "/home/work/chenyo/myPaper/model/LLM-Research/ZhipuAI/glm-4-9b-chat"
    llm = ChatGLM4_LLM(chatGLM4_path)
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
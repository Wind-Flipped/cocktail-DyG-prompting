from typing import Any, List
from api import LLM_API

class LLM():
    history: List = []

    def __init__(self, url="https://openrouter.ai/api/v1/chat/completions",
                 api_key="",
                 model_name="",
                 **kwargs):

        self.api_llm = LLM_API(url=url, api_key=api_key, model_name=model_name, **kwargs)
        self.model_name = model_name
        self.history = []
        print("Successfully get remote api for LLM ")

    def clear_history(self):
        """
        Clears the history used for processing prompts.
        """
        self.history = []

    def __call__(self, prompt: str):
        self.history.append({"role": "user", "content": prompt})
        # get api for LLMs
        response = self.api_llm.getLLMoutput(self.history)

        if response != None:
            self.history.append({"role": "assistant", "content": response})
            return response
        else:
            print("Some error occur in apis")
            return None

    @property
    def _llm_type(self) -> str:
        return self.model_name




if __name__ == '__main__':
    llm = LLM(model_name="mistralai/mistral-7b-instruct")
    a = "1 + 1 = ?"
    print(f"{llm(a)}")
    print("--------------------------")
    a = "What day is it today?"
    print(f"{llm(a)}")
    print("--------------------------")
    a = "How many interactions do we have before this interaction?"
    print(f"{llm(a)}")
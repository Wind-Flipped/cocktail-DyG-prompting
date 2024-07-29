import time
import urllib.request
import json
import os
import ssl
import requests

class ApiLLM():
    url : str
    api_key : str

    def __init__(self, url, api_key, maxtokens=512, temperature=0.0001,
                 top_p=0.9, repetition_penalty=1.1, safe_https=True):
        self.url = url
        self.api_key = api_key
        self.maxtokens = maxtokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        if safe_https:
            self.allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.

    def allowSelfSignedHttps(self, allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    # Request data goes here
    # The example below assumes JSON formatting which may be updated
    # depending on the format your endpoint expects.
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
    def getLLMoutput(self, messages):
        data =  {
          "messages": messages,
          "max_tokens": self.maxtokens,
          "temperature": self.temperature,
          "top_p": self.top_p,
          "repetition_penalty": self.repetition_penalty
        }

        body = str.encode(json.dumps(data))

        url = self.url
        # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
        api_key = self.api_key
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)

            result = response.read()

            result_str = result.decode('utf-8')

            result_json = json.loads(result_str)

            content = result_json["choices"][0]["message"]["content"]

            return content
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))
            return None

class LLM_API(ApiLLM):
    def __init__(self, url, api_key, model_name="meta-llama/llama-3.1-8b-instruct",
                 maxtokens=512, temperature=0.0001,
                 top_p=0.9, repetition_penalty=1.1):
        super(LLM_API, self).__init__(url=url, api_key=api_key, maxtokens=maxtokens, temperature=temperature,
                                           top_p=top_p, repetition_penalty=repetition_penalty, safe_https=False)
        self.model_name = model_name

    def getLLMoutput(self, messages):
        num = 100
        while num > 0:
            num -= 1
            try:
                response = requests.post(
                    url=self.url,
                    headers={
                        "Authorization": self.api_key,
                    },
                    data=json.dumps({
                        "model": self.model_name,  # Optional
                        "messages": messages,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "repetition_penalty": self.repetition_penalty,
                        "max_token": self.maxtokens,
                    })
                )

                reply = json.loads(response.content.decode('utf-8'))['choices'][0]['message']['content']
                return reply
            except Exception as e:
                reply = ""
                print("response---------------")
                print(response)
                print("Exception--------------")
                print(e)
                time.sleep(1)
                return reply



if __name__ == '__main__':
    url = "https://Phi-3-small-8k-instruct-igoal.eastus2.models.ai.azure.com/v1/chat/completions"
    api_key = "o6HUpXMyfPwiXv7Hu0CqLVyC5kWXwm7p"
    api = ApiLLM(url=url, api_key=api_key)
    message = [{"role" : "user", "content": "Who are you?"}]

    print(api.getLLMoutput(messages=message))
    print("--------------------")
    message = [{"role": "user", "content": "1 + 1 =?"}]
    print(api.getLLMoutput(messages=message))
# Import Libraries
import requests
from openai import OpenAI
# Define the headers for the request

class FSLModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.headers = None
        self.url = None
        self.headers = None
        self.api = None
        self.load_model()
    
    ####---Few-Shot-Learning---####
    def load_model(self):

        if self.model_name == 'llama_31_8b' or self.model_name == 'llama_31_70b':
            #huggingface_login()  # Call the login function (requires token in environment)
            # Define the API URL and your DeepInfra API Token
            '''
            self.url = "https://api.deepinfra.com/v1/openai/chat/completions"
            api_token = "xxx"
            self.headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_token}"
                    }
            '''
            self.api = 'xxx'
            if self.model_name == 'llama_31_8b':
                self.llm_name = "llama-v3.1-8b-instruct" #meta-llama/Meta-Llama-3.1-70B-Instruct
            elif self.model_name == 'llama_31_70b':
                self.llm_name = "llama-v3.1-70b-instruct"
            else:
                print('No Model')
            
        elif self.model_name == 'gpt4':
            self.llm_name= "gpt-4o"
            self.api = 'xxx'
        else:
            raise ValueError("The input model is not available.")
        
    def generating(self, prompt_1, prompt_2, few_shot_json, requested_translation):
        data_input = [
                {
                    "role": "system",
                    "content": prompt_1 + few_shot_json
                },
                {
                    "role": "user",
                    "content": prompt_2 + requested_translation
                }
        ]
        if self.model_name == 'gpt4':
            client = OpenAI(
                        api_key=self.api,
                        #base_url = "https://inference.finetunedb.com/v1"
                    )
            response = client.chat.completions.create(model=self.llm_name, messages=data_input)

        else: #this for llama
            client = OpenAI(
                        api_key=self.api,
                        base_url = "https://inference.finetunedb.com/v1"
                    )
            response = client.chat.completions.create(model=self.llm_name, messages=data_input)
                   
        return response

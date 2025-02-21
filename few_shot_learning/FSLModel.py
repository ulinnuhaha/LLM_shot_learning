# Import Libraries
import os
from openai import OpenAI


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
             Define the API Token
            self.api_key = os.getenv("FINETUNEDB_API_KEY")  # Use environment variable for API key #self.api = 'xxx'
            if not self.api_key:
                raise ValueError("API key not found in environment variables.")

            if self.model_name == 'llama_31_8b':
                self.llm_name = "llama-v3.1-8b-instruct" #meta-llama/Meta-Llama-3.1-70B-Instruct
            elif self.model_name == 'llama_31_70b':
                self.llm_name = "llama-v3.1-70b-instruct"
            else:
                print('No Model')
            
        elif self.model_name == 'gpt4':
            self.api_key = os.getenv("OPENAI_API_KEY")  # Use environment variable for API key # self.api = 'xxx'
            if not self.api_key:
                raise ValueError("OpenAI API key not found in environment variables.")
            self.llm_name= "gpt-4o"
           
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
        # Performing inferencing from LLM
        if self.model_name == 'gpt4':
            client = OpenAI(
                        api_key=self.api_key,
                        #base_url = "https://inference.finetunedb.com/v1"
                    )
            response = client.chat.completions.create(model=self.llm_name, messages=data_input)

        else: #this for llama
            client = OpenAI(
                        api_key=self.api_key,
                        base_url = "https://inference.finetunedb.com/v1"
                    )
            response = client.chat.completions.create(model=self.llm_name, messages=data_input)
                   
        return response

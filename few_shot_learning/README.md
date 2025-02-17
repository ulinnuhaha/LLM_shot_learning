## Few-Shot Learning
In this code we use API key from OpenAI to run Few-Shot Learning using GPT and FinetuneDB to run Few-Shot Learning using Llama.
To set the API key, Set the FINETUNEDB_API_KEY and OPENAI_API_KEY environment variables before running the script.
```
export DEEPINFRA_API_KEY="your_deepinfra_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```
Here, we use GPT-4 version for GPT and Llama-3.1-8B version for LLama
To change the API provider and the LLMs version of GPT and Llama you can go to:
```
FSLModel.py
```

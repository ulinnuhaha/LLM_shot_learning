## Zew-Shot Learning
In this code we use API key from OpenAI to run Zero-Shot Learning using GPT and FinetuneDB to run Zero-Shot Learning using Llama.
To set the API key, Set OPENAI_API_KEY and  the FINETUNEDB_API_KEY environment variables before running the script.
```
export OPENAI_API_KEY="your_openai_api_key"
export FINETUNEDB_API_KEY="your_finetunedb_api_key"
```
Here, we use GPT-4 version for GPT and Llama-3.1-8B version for LLama
To change the API provider and the LLMs version of GPT and Llama you can go to:
```
ZSLModel.py
```

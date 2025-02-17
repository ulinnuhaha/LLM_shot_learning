# Shot Learning 
This repository focuses on exploring few-shot learning and zero-shot learning techniques for machine translation (MT) using Large Language Models (LLMs). We leverage models like GPT and Llama to perform shot learning via their respective APIs.

## Zero-Shot Learning
If you want to use zero-shot learning, run the following command:
```
python run_trans.py --model_name_or_path ./pretrained_model/nllb_tr_ch
```
## Few-Shot Learning
If you want to use few-shot learning, run the following command:
```
python run_trans.py --model_name_or_path ./pretrained_model/nllb_tr_ch
```
## Evaluation
Before performing the evaluation, ensure you have prepared a parallel sentences dataset.
To run the evaluation, use the evaluation.ipynb script:
```
evaluation.ipynb
```

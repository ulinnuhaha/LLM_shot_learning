# Shot Learning 
This repository focuses on exploring few-shot learning and zero-shot learning techniques for machine translation (MT) using Large Language Models (LLMs). We leverage models like GPT and Llama to perform shot learning via their respective APIs. To run the code in this repository, you should already have a parallel sentence dataset in CSV format

## Zero-Shot Learning
If you want to use zero-shot learning, run the following command:
```
python zsl_main.py \
  --model_name llama_31_8b \
  --target_lang italian \
  --test_data ./data_dir/test_data \
  --batch_size 25 \
  --save_dir ./save_results
```
If you want to change the API provider and the LLM version please go to `zero_shot_learning` repository.

## Few-Shot Learning
If you want to use few-shot learning, run the following command:
```
python fsl_main.py \
  --model_name llama_31_8b \
  --dataset ./data_dir/dataset \
  --target_lang italian \
  --test_data ./data_dir/test_data \
  --batch_size 25 \
  --save_dir ./save_results
```
If you want to change the API provider and the LLM version please go to `few_shot_learning` repository.

## Evaluation
The output data of both learning schemes will be in JSON format as follows:

`{"translations": [
   {
   "Source_lang": "xxx....",
   "Target_lang": "xxx..."
   }
   ...
   {
   "Source_lang": "xxx...",
   "Target_lang": "xxx..."
 }
]}`.

Before performing the evaluation, ensure you have prepared a parallel sentences dataset.
To run the evaluation, use the evaluation.ipynb script:
```
evaluation.ipynb
```

#!/usr/bin/env python
# coding: utf-8
# import the libraries
import os
import pandas as pd
import json
import argparse
from zero_shot_learning import ZSLModel

# use argparse to let the user provides values for variables at runtime
def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
        type=str, required=True, help='Load a LLM as model for few-shot learning')
    parser.add_argument('--batch_size', 
        type=int, default=20, help='Number of request sentences per batch of target translation')
    parser.add_argument('--save_dir', 
        type=str, required=True, help='Directory for saving experimental results')
    args = parser.parse_args()
    return args

#!python zsl_main.py --model_name gpt4 --batch_size 20 --save_dir ./save_results

def main():
    #create the configuration class
    args=DataTestingArguments() #call the arguments
    ####---Load Dataset---####
    dataset = pd.read_csv(f"Trip_adv_ita_mono.csv")
    dataset = dataset[24100:]
    # Convert DataFrame to JSON format for few-shot examples
    def json_conv(data):
        return {"translations": data.to_dict(orient='records')}

    # Set the LLMs
    zsl_model = ZSLModel(args.model_name)
    index_batch = 1205
    for batch_start in range(0, len(dataset), args.batch_size):       
        # Select a rotating subset of few-shot examples
        zero_shot_examples = dataset.iloc[batch_start:batch_start + args.batch_size] 
         # Increment for the next batch

        # Convert DataFrame to JSON format
        #zero_shot_examples = json_conv(zero_shot_examples)        
        # Construct the prompt for the model
        #zero_shot_examples=json.dumps(zero_shot_examples, ensure_ascii=False) 


        # Convert DataFrame to a list in the desired format
        #formatted_few_shot_examples = [f"[texts: {row['italian']}" for _, row in few_shot_examples.iterrows()]
        formatted_shot_examples = [f"[{row['italian']}]" for _, row in zero_shot_examples.iterrows()]

        # make it to string
        formatted_shot_examples = ", ".join(formatted_shot_examples)
        # print(formatted_shot_examples)

        # Prompt templates
        prompt_1 = ("Please review and correct the following Italian texts, ensuring proper Italian grammar, syntax, and style. Focus on clarity and accuracy while preserving the original meaning:\n"
                    )
        prompt_2 = (
                    f"\nReturn the corrected version in the original format, as a list (e.g., [xxx], [xxx], ..., [xxx]). Do not include any additional explanations or the original texts."
                    )
        
        # Generate translation using LLMs with API
        generated_translation = zsl_model.generating(prompt_1, formatted_shot_examples, prompt_2)
        
        # If the response is successful
        #if generated_translation.status_code == 200:
        try:
            response_json = generated_translation.json()
            print(f"Response JSON for batch {index_batch} ")

            # prepraing set the output into json file
            if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
            output_path = os.path.join(args.save_dir, f'corrected_Trip_adv_ita_mono_{args.model_name}_batch_{index_batch}.json')
                
            # Save the JSON response for this batch
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(response_json, json_file, ensure_ascii=False, indent=4)
                
            print(f"Saved batch {index_batch} to {output_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print("Error parsing the translation output.")
                
        #else:
        #    print(f"Error: {generated_translation.status_code}, {generated_translation.text}")

        index_batch += 1    
if __name__ == "__main__":
    main()

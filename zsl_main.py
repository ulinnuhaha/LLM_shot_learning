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
    parser.add_argument('--target_lang', 
        type=str, required=True, help='The language of translation target')
    parser.add_argument('--test_data', 
        type=str, required=True, help='Directory of the CSV test data file')
    parser.add_argument('--batch_size', 
        type=int, default=20, help='Number of request sentences per batch of target translation')
    parser.add_argument('--save_dir', 
        type=str, required=True, help='Directory for saving the results')
    args = parser.parse_args()
    return args

def main():
    #create the configuration class
    args=DataTestingArguments() #call the arguments
    
    ####---Load Dataset---####
    dataset = pd.read_csv(f"{args.test_data}.csv")
    
    #dataset = dataset[24100:]
    # Set the source and target languages
    if args.target_lang == 'language_1':
        dataset['language_1'] = " " # empty the target language in test data as we want to predict it
        source_lang = 'language_2'
        target_lang = 'language_1'
        path_f = 'ladin2italian' # Please change the name of the path for the result directory
    else:
        dataset['language_2'] = " " # empty the target language in test data as we want to predict it
        source_lang = 'language_1'
        target_lang = 'language_2'
        path_f = 'italian2ladin' # Please change the name of the path for the result directory
    # Convert DataFrame to JSON format for few-shot examples
    def json_conv(data):
        return {"translations": data.to_dict(orient='records')}

    # Translation loop with rotating few-shot examples
    
    # Set the LLM
    zsl_model = ZSLModel(args.model_name)
    index_batch = 0
    for batch_start in range(0, len(dataset), args.batch_size):       
       # Get the batch of test data
        data_batch = dataset.iloc[batch_start:batch_start + args.batch_size]        
        
        # Convert DataFrame to JSON format
        requested_translation = json_conv(data_batch[[source_lang, target_lang]])
        
        # Construct the prompt for the model
        requested_translation = json.dumps(requested_translation, ensure_ascii=False)

        # Write the prompts
        prompt_1 = (f"You are machine translation between {source_lang} and {target_lang}:\n"
                    )
        prompt_2 = (
                    f"\n Please provide the translation of the following {len(data_batch)} entries in the JSON format, filling the empty '{target_lang}' fields for each entry. "
                    "Do not include any additional explanations or text.\n"
                    )

        # Generate translation using LLMs with API
        generated_translation = zsl_model.generating(prompt_1, prompt_2, requested_translation)
        
        # If the response is successful
        #if generated_translation.status_code == 200:
        try:
            response_json = generated_translation.json()
            print(f"Response JSON for batch {index_batch} ")

            # prepraing set the output into json file
            if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
            output_path = os.path.join(args.save_dir, path_f, f'translation_{args.model_name}_for_{args.target_lang}_size of_{args.batch_size}_batch_{index_batch}.json')
                
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

#!/usr/bin/env python
# coding: utf-8
# import the libraries
import os
import pandas as pd
import json
import argparse
from few_shot_learning import FSLModel

# use argparse to let the user provides values for variables at runtime
def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
        type=str, required=True, help='Load a LLM as model for few-shot learning')
    parser.add_argument('--dataset_dir', 
        type=str, required=True, help='Directory of the dataset files')
    parser.add_argument('--target_lang', 
        type=str, required=True, help='Directory of the dataset files')
    parser.add_argument('--test_data', 
        type=str, required=True, help='Directory of the testing data files')
    parser.add_argument('--batch_size', 
        type=int, default=20, help='Number of request sentences per batch of target translation')
    parser.add_argument('--save_dir', 
        type=str, required=True, help='Directory for saving experimental results')
    args = parser.parse_args()
    return args

def main():
    #create the configuration class
    args=DataTestingArguments() #call the arguments
    
    ####---Load Dataset---####
    train_data = pd.read_csv(os.path.join(args.dataset_dir, f"ita2lad_dataset.csv"))
    test_data = pd.read_csv(os.path.join(args.dataset_dir, f"{args.test_data}.csv"))
    # Capitalize only the first character of each column name
    train_data.columns = train_data.columns.str.capitalize()
    test_data.columns = test_data.columns.str.capitalize()

    test_data = test_data #[:4]

    # Set the source and target languages
    if args.target_lang == 'italian':
        test_data['Italian'] = " "
        source_lang = 'Ladin'
        target_lang = 'Italian'
        path_f = 'ladin2italian'
    else:
        test_data['Ladin'] = " "
        source_lang = 'Italian'
        target_lang = 'Ladin'
        path_f = 'italian2ladin'
    
    # Convert DataFrame to JSON format for few-shot examples
    def json_conv(data):
        return {"translations": data.to_dict(orient='records')}
    
    # Translation loop with rotating few-shot examples
    rotation_index = 0  # Initialize a cumulative rotation index

    # Set the LLMs
    fsl_model = FSLModel(args.model_name)
    index_batch = 0
    train_data = train_data[[source_lang, target_lang]]
    for batch_start in range(0, len(test_data), args.batch_size):
        # Get the batch of test data
        test_batch = test_data.iloc[batch_start:batch_start + args.batch_size]
        
        # Select a rotating subset of few-shot examples
        few_shot_examples = train_data.iloc[rotation_index:rotation_index + (args.batch_size*2)] 

        # Wrap around if reaching the end of train_data
        if rotation_index + (args.batch_size*2) >= len(train_data):
            rotation_index = (rotation_index + (args.batch_size*2)) % len(train_data)
        else:
            rotation_index += (args.batch_size*2) # Increment for the next batch

        # Convert DataFrame to JSON format
        few_shot_json = json_conv(few_shot_examples)
        requested_translation = json_conv(test_batch[[source_lang, target_lang]])
        
        # Construct the prompt for the model
        few_shot_json=json.dumps(few_shot_json, ensure_ascii=False) 
        requested_translation = json.dumps(requested_translation, ensure_ascii=False)

        # Write the prompts
        prompt_1 = ("Here are examples of translations in a JSON format between Italian and Ladin with the Val Badia variant:\n"
                    )
        prompt_2 = (
                    f"\n Please provide the translation of the following {len(test_batch)} entries in the JSON format, filling the empty '{target_lang}' fields for each entry. "
                    "Do not include any additional explanations or text.\n"
                    )
        
        '''
        # Convert DataFrame to a list in the desired format
        formatted_test_batch = [f"[source_lang: {row[source_lang]}, ladin: {row[target_lang]}]" for _, row in test_batch.iterrows()]
        formatted_few_shot_examples = [f"[source_lang: {row[source_lang]}, target_lang: {row[target_lang]}]" for _, row in few_shot_examples.iterrows()]
        # make it to string
        formatted_test_batch = ", ".join(formatted_test_batch)
        formatted_few_shot_examples = ", ".join(formatted_few_shot_examples)

        # Prompt templates
        prompt_1 = ("Here are examples of translations between Italian and Ladin with the Val Badia variant:\n"
                    )
        prompt_2 = (
                    f"\n Please provide the translation of the following {len(test_batch)} entries by filling the empty '{target_lang}' fields for each entry. "
                    "Do not include any additional explanations or text.\n"
                    )
        '''

        # Generate translation using LLMs with API
        generated_translation = fsl_model.generating(prompt_1, prompt_2, few_shot_json, requested_translation)
        
        # If the response is successful
        #if generated_translation.status_code == 200:
        try:
            response_json = generated_translation.json()
            print(f"Response JSON for batch {index_batch} ")

            # prepraing set the output into json file
            if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
            output_path = os.path.join(args.save_dir, path_f, f'translation_{args.model_name}_{args.test_data}_size of_{args.batch_size}_batch_{index_batch}.json')
                
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

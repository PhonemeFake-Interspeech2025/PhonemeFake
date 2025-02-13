import os
from openai import OpenAI
from tqdm import tqdm
import argparse

def process_directory(client, root_dir):
    # Traverse the directory and process .txt files
    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files):
            if file.endswith('_text.txt'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                # Analyze and generate the output
                result = analyze_and_replace(client, text)

                # Save the result to a new file
                output_file = os.path.join(subdir, f"{file}_modified_gpt4o.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                    

def analyze_and_replace(client, input_text):
    response = client.chat.completions.create(
    messages=[
            {"role": "system", "content": "You are a tool designed to generate DeepFake news by modifying sentences. Your task is to identify the most crucial word in a given sentence and replace it with a word that changes the sentence's meaning to its opposite. Provide your response in the following format: Old word: [insert identified word] New word: [insert replacement word]"},
            {"role": "user", "content": input_text}
    ],
    model="gpt-4o",
    )
    
    output_text = response.choices[0].message.content.replace("\"", "").split()
    # print(output_text)
    old_word = output_text[2]
    new_word = output_text[5]
    input_split= input_text.split(old_word)
    if len(input_split)>2:
        first_split = input_split[0]
        second_split = ""
        for i in range(1,len(input_split)):
            second_split+=input_split[i]
        input_split = [first_split, second_split]
        print("preprocessing the input txt, multiple occurences")
    modified_text = input_split[0]+"|"+new_word+"|"+old_word+"|"+input_split[-1]
    return modified_text


def main(args):
    print("Initializing...")
    os.environ['OPENAI_API_KEY'] = args.api_key
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
    )
    root_directory = args.root_directory
    process_directory(client, root_directory)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a directory of text files with GPT-4")
    parser.add_argument(
        "--root_directory",
        type=str,
        required=False,
        help="Root directory containing the text files to process",
        default = "/home/ob3942/repos/PhonemeFake/dataset1"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        help="OpenAI API key",
        #default = ""
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=False,
        help="LLM to process the text",
        default = "GPT4o"
    )
    args = parser.parse_args()
    main(args)

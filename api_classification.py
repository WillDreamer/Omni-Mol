# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
from helpers import load_json, save_json
from tqdm import tqdm
import argparse


def process(raw):
    instruction = raw["instruction"]
    input_selfies = raw["input"]
    instruction = instruction + " " + input_selfies
    gt = raw["output"]
    
    return instruction, gt


def main(args):
    client = OpenAI(api_key="sk-5c4c695479d74a8cb8744f1e419c85fe", base_url="https://api.deepseek.com")
    data_file = load_json(args.data_path)
    data_file = [raw for raw in data_file if raw["metadata"]["split"] == "test"]
    base_prompt = " Output True or False as your answer only!"
    all_answer = []
    pbar = tqdm(total=len(data_file))
    for raw in data_file:
        instruction, gt = process(raw)
        instruction += base_prompt
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": instruction},
            ],
            stream=False
        )
        
        all_answer.append({
            "instruction": instruction,
            "response": response.choices[0].message.content,
            "gt": gt
        })
        pbar.update(1)

        pbar.write(f"\ninstruction: {instruction}\nresponse: {response.choices[0].message.content}\ngt:{gt}\n")
    
    
    save_json(all_answer, args.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    
    args = parser.parse_args()
    main(args)

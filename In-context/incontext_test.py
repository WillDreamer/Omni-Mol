import json
from openai import OpenAI 
import re
from tqdm import tqdm
import time


total_sleep = int(60 * 60 * 3)
start_time = time.time()
with tqdm(total=total_sleep, desc="休眠进度", unit="秒") as pbar:
    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_sleep:
            break
        time.sleep(1)
        pbar.update(1)
print('Start!')

client = OpenAI(api_key="sk-15723287f95b42448a3ea4ef84a2bf85", base_url="https://api.deepseek.com")
task = 'forward'
save_name = f'{task}_in_context_answer.json'
file_name = f"{task}_in_context_data.json"
with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

in_context_data = []

for index, item in enumerate(tqdm(data)):
    input_mol = item['input_mol']
    output_mol = item["output_mol"]
    instruction = item["instruction"]
    top_train_samples = item["top_train_samples"]

    prompt = "Context: \n " 
    prompt += top_train_samples
    prompt += f"Question: The input is {input_mol} \n {instruction} \n "
    prompt += "Directly output the final SELFIES answer, do NOT output any other words."

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    response = response.choices[0].message.content

    in_context_data.append({
        "prompt": prompt,
        "gt":output_mol,
        "pred":response
    })

# 保存 in-context 数据到文件（可选）
with open(save_name, "w", encoding="utf-8") as f:
    json.dump(in_context_data, f, ensure_ascii=False, indent=2)

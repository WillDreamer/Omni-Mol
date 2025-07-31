from datasets import Dataset, load_dataset
from tqdm import tqdm
import json
from typing import Dict, Tuple, Optional
import selfies as sf
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

data_path = "/wanghaixin/OmniMol/Molecule-oriented_Instructions/evaluate/forward_reaction_prediction.json"
# data_path = '/wanghaixin/OmniMol/Molecule-oriented_Instructions/train/reagent_prediction.json'
# data_path = '/wanghaixin/OmniMol/Molecule-oriented_Instructions/train/retrosynthesis.json'
save_name = "forward_in_context_data.json"

def sf_encode(selfies):
    """Returns the smiles representation, returns None if it
    cannot be converted by selfies.decoder

    Args:
        selfies (str): input selfies

    Returns:
        str | None: smiles or None
    """
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception as e:
        print(f"Convert {selfies} to SMILES failed, {e}")
        return None

def parse_model_answer(answer_text: str) -> Optional[Dict[str, str]]:

    status_dict = {}

    output_selfies = answer_text.strip() # selfies
    out_smiles = sf_encode(output_selfies)
    
    if out_smiles is not None:
        return out_smiles
    else:
        return None

dataset = load_dataset("json", data_files=data_path)  
train_dataset = dataset["train"].filter(lambda x: x["metadata"]["split"] == "train")
test_dataset = dataset["train"].filter(lambda x: x["metadata"]["split"] == "test")

# 预先遍历 train_dataset，计算每个样本的 SMILES 和指纹
train_examples = []  # 保存有效的训练样本及其信息
train_fps = []       # 保存对应的 Morgan 指纹

print("Precomputing fingerprints for train_dataset...")
for sample in tqdm(train_dataset):
    train_output = sample['input']
    # train_output = train_output.split('>>')[0]
    train_smiles = parse_model_answer(train_output)
    if train_smiles is None:
        continue
    mol = Chem.MolFromSmiles(train_smiles)
    if mol is None:
        continue
    fp = AllChem.GetMorganFingerprint(mol, 2)
    train_examples.append(sample)
    train_fps.append(fp)

in_context_data = []

for batch in tqdm(test_dataset):
    instruction = batch['instruction']
    input_mol = batch['input']
    output_mol = batch['output']

    # input_mol_ = input_mol.split('>>')[0]
    # test_smiles = parse_model_answer(input_mol_)
    test_smiles = parse_model_answer(input_mol)
    if test_smiles is None:
        continue
    test_mol = Chem.MolFromSmiles(test_smiles)
    if test_mol is None:
        continue
    test_fp = AllChem.GetMorganFingerprint(test_mol, 2)
    # 计算与每个 train 样本的相似度
    scores = []
    for fp in train_fps:
        score = DataStructs.TanimotoSimilarity(test_fp, fp)
        scores.append(score)
    # 找出相似度最高的前5个索引
    top_five_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    top_train_samples = [train_examples[i] for i in top_five_indices]
    top_scores = [scores[i] for i in top_five_indices]

    ## 处理top_train_samples成模板，
    template = ""
    for sample in top_train_samples:
        template += f"Question: The input is {sample['input']} \n {sample['instruction']} \n "
        template += f"Answer: {sample['output']}\n"
        template += "---\n"  # 用分隔符区分不同样本
    
    in_context_data.append({
        "input_mol": input_mol,
        "output_mol":output_mol,
        "instruction":instruction,
        "top_train_samples": template,
        "top_scores": top_scores
    })

# 保存 in-context 数据到文件（可选）
with open(save_name, "w", encoding="utf-8") as f:
    json.dump(in_context_data, f, ensure_ascii=False, indent=2)

    

    
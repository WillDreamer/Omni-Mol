import re

import json

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, LlamaTokenizer, BertTokenizerFast
from rdkit import DataStructs
from rdkit import Chem
import selfies as sf
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from sklearn import metrics

from Levenshtein import distance as lev

import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

def evaluate_caption(predictions, targets, tokenizer, text_trunc_length=512):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length)
        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length)

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        try:
            mscore = meteor_score([gt_tokens], out_tokens)
            meteor_scores.append(mscore)
        except:
            continue


    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score

def evaluate_naming_description(data, tokenizer, path, task, text_trunc_length=512):
    pred = [line['prediction'].strip() for line in data]
    target = [line['target'].strip() for line in data]
    
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
        evaluate_caption(pred, target, tokenizer, text_trunc_length)
        
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score

def evaluate_reaction(data, path, task):
    pred_list, test_list = [], []

    for d in data:
        curr_pred = d['prediction'].replace(" ", "").strip()
        pred_mol = Chem.MolFromSmiles(curr_pred)
        curr_tgt = d['target'].replace(" ", "").strip()
        tgt_mol = Chem.MolFromSmiles(curr_tgt)
        
        canonical_pred = Chem.MolToSmiles(pred_mol, isomericSmiles=False, canonical=True) if pred_mol else None
        canonical_tgt = Chem.MolToSmiles(tgt_mol, isomericSmiles=False, canonical=True) if tgt_mol else None
        
        if canonical_tgt is None or canonical_pred is None:
            continue
        
        pred_list.append(canonical_pred)
        test_list.append(canonical_tgt)
    
    references_list = []
    hypotheses_list = []
    outputs_rdkit_mols = []
    levs = []
    num_exact = 0
    
    for pred, test in zip(pred_list, test_list):
        pred_tokens = [c for c in pred]
        test_tokens = [c for c in test]
        references_list.append([test_tokens])
        hypotheses_list.append(pred_tokens)
        
        try:
            m_out = Chem.MolFromSmiles(pred)
            m_gt = Chem.MolFromSmiles(test)
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1
            outputs_rdkit_mols.append((m_gt, m_out))
        except:
            continue
        levs.append(lev(pred, test))
    
    blue_score = corpus_bleu(references_list, hypotheses_list)
    
    # Calculate similarities
    MACCS_sims, morgan_sims, RDK_sims = [], [], []
    morgan_r = 2
    
    for gt_m, ot_m in outputs_rdkit_mols:
        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), 
                                                           MACCSkeys.GenMACCSKeys(ot_m)))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), 
                                                         Chem.RDKFingerprint(ot_m)))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m, morgan_r),
                                                         AllChem.GetMorganFingerprint(ot_m, morgan_r)))
    
    results = {
        'BLEU': blue_score,
        'Levenshtein': sum(levs)/len(levs),
        'Exact Match': num_exact/len(test_list),
        'MACCS Similarity': np.mean(MACCS_sims),
        'RDK Similarity': np.mean(RDK_sims),
        'Morgan Similarity': np.mean(morgan_sims),
    }
    
    return results

def evaluate_property(data, path, task):
    pred_list = []
    tgt_list = []
    
    for d in data:
        try:
            pred_list.append(float(d['prediction']))
            tgt_list.append(float(d['target']))
        except:
            continue
    
    mae = metrics.mean_absolute_error(tgt_list, pred_list)
    
    return mae

def evaluate_task(task, file=None, tokenizer=None):
    """Main function to evaluate a specific task
    
    Args:
        task (str): Task to evaluate ('desc', 'forward', 'retro', 'property', 'naming')
        file (str): File path to the predictions
        tokenizer: Tokenizer to use for text-based tasks
    """
    if file is None:
        file = "/all_checkpoints/temp/lightning_logs/version_0/predictions.txt"
    
    with open(file, "r") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    
    if task in ["desc", "naming"]:
        if tokenizer is None:
            raise ValueError("Tokenizer required for description and naming tasks")
        return evaluate_naming_description(data, tokenizer, file, task)
    
    elif task in ["forward", "retro"]:
        return evaluate_reaction(data, file, task)
    
    elif "prop" in task:
        return evaluate_property(data, file, task)
    
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model predictions for various chemistry tasks')
    
    parser.add_argument('--task', type=str, required=True,
                      choices=['desc', 'forward', 'retro', 'property', 'naming'],
                      help='Task to evaluate')
    
    parser.add_argument('--path', type=str, 
                      default="/hub_data5/jinyoungp/all_checkpoints/newLLaMo_epoch3_epoch3_ft_",
                      help='Base path to the model checkpoints')
    
    args = parser.parse_args()
    
    # Initialize tokenizer if needed for the task
    tokenizer = None
    if args.task in ["desc", "naming"]:
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", 
            use_fast=False
        )
    
    # Run evaluation
    results = evaluate_task(
        task=args.task,
        file=args.path,
        tokenizer=tokenizer
    )
    
    # Print results
    print(f"\nResults for {args.task} task:")
    if isinstance(results, dict):
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    else:
        print(results)


'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import os
import re
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
import textdistance
RDLogger.DisableLog('rdApp.*')
import selfies as sf
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, r2_score
from loggers import WrappedLogger
from helpers import save_json, load_json

logger = WrappedLogger(__name__)


def sf_encode(selfies: str) -> str | None:
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
        logger.warning(f"Convert {selfies} to SMILES failed, {e}")
        return None

def convert_to_canonical_smiles(smiles: str) -> str | None:
    """Convert to canonical SMILES

    Args:
        smiles (str): input smiles string

    Returns:
        str | None: canonical smiles or None
    """
    if smiles is not None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
            return canonical_smiles
        
    return None

def build_evaluate_tuple(result:dict):
    result["pred_smi"] = convert_to_canonical_smiles(sf_encode(result["pred"]))
    result["gt_smi"] = convert_to_canonical_smiles(sf_encode(result["gt"]))
    
    return result


def calc_fingerprints(input_file: str, save_path=None, morgan_r: int=2, eos_token='<|end|>'):
    outputs = []
    ans_file = load_json(input_file)
    
    for result in ans_file:
        result['pred'] = result['pred'].split(eos_token)[0]
        result = build_evaluate_tuple(result)
        gt_m = Chem.MolFromSmiles(result['gt_smi'])
        try:
            ot_m = Chem.MolFromSmiles(result['pred_smi'])
            if ot_m is not None:
                outputs.append((result['prompt'], gt_m, ot_m))
        except Exception as e:
            logger.warning(f"Unable to build molecule {result['pred']}")
        
    validity_score = float(len(outputs)) / float(len(ans_file))
    
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    for _, gt_m, ot_m in outputs:
        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    # np.sum(MACCS_sims) / len(outputs) + bad_mols
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    
    logger.info(f'validity: {validity_score}')
    logger.info(f'Average MACCS Similarity: {maccs_sims_score}')
    logger.info(f'Average RDK Similarity: {rdk_sims_score}')
    logger.info(f'Average Morgan Similarity: {morgan_sims_score}\n')
    new_metrics = [
        {"Validity": float(validity_score)},
        {"Average MACCS Similarity": float(maccs_sims_score)},
        {"Average RDK Similarity": float(rdk_sims_score)},
        {"Average Morgan Similarity": float(morgan_sims_score)}
    ]
    # if save_path is not None:
    #     save_json(metrics_list, save_path)
    if save_path is not None:
        if os.path.exists(save_path):
            existing_data = load_json(save_path)
            if isinstance(existing_data, list):
                combined_data = existing_data + [new_metrics]
            else:
                combined_data = [new_metrics]
        else:
            combined_data = [new_metrics]
        save_json(combined_data, save_path)
        
    return new_metrics
        
        
def calc_mol_trans(input_file, metric_path=None, eos_token='<|end|>'):
    outputs = []
    ans_file = load_json(input_file)
    bad_mols = 0
    for result in ans_file:
        result['pred'] = result['pred'].split(eos_token)[0]
        result = build_evaluate_tuple(result)
        if result['pred_smi'] is not None:
            outputs.append((result['prompt'], result['gt'], result['pred'], result['gt_smi'], result['pred_smi']))
        else:
            bad_mols += 1

    references_self_tokens = []
    hypotheses_self_tokens = []
    levs_self = []
    levs_smi = []
    num_exact = 0
    for des, gt_self, ot_self, gt_smi, ot_smi in outputs:
        gt_self_tokens = [c for c in gt_self]
        out_self_tokens = [c for c in ot_self]
        references_self_tokens.append([gt_self_tokens])
        hypotheses_self_tokens.append(out_self_tokens)
        levs_self.append(lev(ot_self, gt_self))
        levs_smi.append(lev(ot_smi, gt_smi))
        m_out = Chem.MolFromSmiles(ot_smi)
        m_gt = Chem.MolFromSmiles(gt_smi)
        try:
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1
        except Exception as e:
            logger.warning(f"Molecule {ot_self}, {e}")
            bad_mols += 1
        
    # BLEU score
    bleu_score_self = corpus_bleu(references_self_tokens, hypotheses_self_tokens)
    logger.info(f'SELFIES BLEU score {bleu_score_self}')
    # Exact matching score
    exact_match_score = num_exact/len(outputs)
    logger.info(f'Exact Match: {exact_match_score}')
    # Levenshtein score
    levenshtein_score_smi = np.mean(levs_smi)
    logger.info(f'SMILES Levenshtein: {levenshtein_score_smi}')
        
    validity_score = (len(ans_file) - bad_mols) / len(ans_file)
    logger.info(f'validity: {validity_score}', )
    metrics_list = [
        {"SELFIES BLEU score": float(bleu_score_self)},
        {"Exact Match": float(exact_match_score)},
        {"SMILES Levenshtein": float(levenshtein_score_smi)},
        {"validity": float(validity_score)}
    ]
    if metric_path is not None:
        save_json(metrics_list, metric_path)
    
    return metrics_list


def compute_mae(eval_result_file: str, metric_path: str, eos_token: str):
    data_dict = {
        "homo_gts": [],
        "homo_preds": [],
        "lumo_gts": [],
        "lumo_preds": [],
        "gap_gts": [],
        "gap_preds": []
    }

    error_count = 0

    with open(eval_result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

        for i, result in enumerate(results):
            pred = result['pred'].split(eos_token)[0]
            gt = result['gt']
            prompt = result['prompt']

            try:
                gt_val = float(gt)
                pred_val = float(pred)
            except:
                error_count += 1
                continue

            # 根据 prompt 判断归属，并将数据存入对应列表
            if "HOMO" in prompt and "LUMO" not in prompt:
                data_dict["homo_gts"].append(gt_val)
                data_dict["homo_preds"].append(pred_val)
            elif "HOMO" not in prompt and "LUMO" in prompt:
                data_dict["lumo_gts"].append(gt_val)
                data_dict["lumo_preds"].append(pred_val)
            elif "HOMO" in prompt and "LUMO" in prompt:
                data_dict["gap_gts"].append(gt_val)
                data_dict["gap_preds"].append(pred_val)
            else:
                raise NotImplementedError("prompt 内容既不包含 HOMO 也不包含 LUMO，或无法识别。")
            f.close()

    # 当各列表中至少有一个值才能进行 MAE 计算，否则可以根据需求做相应处理
    homo_err = mean_absolute_error(data_dict["homo_gts"], data_dict["homo_preds"]) if data_dict["homo_gts"] else None
    lumo_err = mean_absolute_error(data_dict["lumo_gts"], data_dict["lumo_preds"]) if data_dict["lumo_gts"] else None
    gap_err  = mean_absolute_error(data_dict["gap_gts"],  data_dict["gap_preds"])  if data_dict["gap_gts"]  else None

    total_gts = data_dict["homo_gts"] + data_dict["lumo_gts"] + data_dict["gap_gts"]
    total_preds = data_dict["homo_preds"] + data_dict["lumo_preds"] + data_dict["gap_preds"]
    average = mean_absolute_error(total_gts, total_preds) if total_gts else None
    print('Final:',)
    print("HOMO MAE:", homo_err)
    print("LUMO MAE:", lumo_err)
    print("GAP MAE:", gap_err)
    print("Average:", average)
    print("Number of skipped items (转换失败):", error_count)

    metrics_list = [
        {"HOMO MAE": homo_err},
        {"LUMO MAE": lumo_err},
        {"GAP MAE": gap_err},
        {"Average": average},
        {"Skipped items": error_count}
    ]

    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)
        f.close()
        
    return metrics_list


def compute_extracted_mae(eval_result_file: str, metric_path: str, eos_token: str):
    data_dict = {
        "gts": [],
        "preds": []
    }
    pattern = r"\d+(?:\.\d+)?"
    invalid_samples = []  # 用于记录匹配失败的样例

    with open(eval_result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

        for i, result in enumerate(results):
            # 需提取数字
            pred = result['pred'].split(eos_token)[0]
            gt = result['gt']

            pred_nums = re.findall(pattern, pred)
            gt_nums = re.findall(pattern, gt)

            # 如果 pred 或 gt 中无法匹配到数字，跳过该条数据并记录下来
            if not pred_nums or not gt_nums:
                invalid_samples.append({
                    "index": i,  # 或者换成你的标识
                    "prompt": result.get("prompt", ""),
                    "pred": pred,
                    "gt": gt
                })
                continue

            # 提取第一个匹配到的数字
            ex_pred = pred_nums[0]
            ex_gt = gt_nums[0]

            data_dict["gts"].append(float(ex_gt))
            data_dict["preds"].append(float(ex_pred))

        f.close()
    # 如果 data_dict 中有可用数据，才进行 MAE 计算
    if len(data_dict["gts"]) > 0 and len(data_dict["preds"]) > 0:
        mae = mean_absolute_error(data_dict["gts"], data_dict["preds"])
    else:
        # 如果全部都无法匹配到数字，可根据需求设置一个默认值或者报错
        mae = None

    print("Average MAE:", mae)

    # 将 MAE 与 invalid_samples 一起写入到最终的 JSON 文件
    metrics_list = [
        {"Average MAE": float(mae) if mae is not None else None},
        {"invalid_samples": invalid_samples}
    ]

    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)
        f.close()
        
    return metrics_list


def compute_extracted_SCF_mae(eval_result_file: str, metric_path: str, eos_token: str, max_examples=5):
    """
    从eval_result_file中读取列表（内部为字典: {'gt':..., 'pred':..., ...}），
    解析提取"SCF Energy"对应的数值，并计算MAE。

    如果某条记录有以下问题，会被跳过：
      1. pred中找不到任何符合正则的数字
      2. gt中找不到任何符合正则的数字
      3. 匹配到数字但转换float时失败 (ValueError)

    统计各问题的出现次数，并且保存若干条出错示例（默认各保留前5条）。

    最终结果写入metric_path，包括：
      - "Average MAE"：平均绝对误差
      - "No pred match"：pred匹配失败的数量
      - "No gt match"：gt匹配失败的数量
      - "Float conversion error"：float转化失败的数量
      - 以及各类问题对应的示例

    Args:
        eval_result_file (str): 输入 JSON 文件路径。
        metric_path (str): 结果输出 JSON 文件路径。
        eos_token (str): 用于截断 pred 的标记。
        max_examples (int): 每种错误类型最多保留多少条示例。
    """

    data_dict = {
        "gts": [],
        "preds": [],
    }

    # 能够匹配负号与科学计数法的正则表达式
    pattern = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

    # 三种失败原因的计数
    no_pred_match = 0
    no_gt_match = 0
    float_conversion_error = 0

    # 用于保存样本示例（只保留部分）
    no_pred_match_samples = []
    no_gt_match_samples = []
    float_conversion_error_samples = []

    if not os.path.exists(eval_result_file):
        print(f"输入文件不存在: {eval_result_file}")
        return

    with open(eval_result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
        for i, result in enumerate(results):
            # 读取并截断pred
            pred = result.get('pred', '').split(eos_token)[0]
            gt = result.get('gt', '')

            # 用正则在字符串中寻找数字
            ex_pred_list = re.findall(pattern, pred)
            ex_gt_list = re.findall(pattern, gt)

            # 检查是否匹配到数字
            if not ex_pred_list:
                no_pred_match += 1
                # 只记录前 max_examples 条示例
                if len(no_pred_match_samples) < max_examples:
                    no_pred_match_samples.append({
                        "index": i,
                        "pred_original": result.get('pred', ''),
                        "gt_original": result.get('gt', ''),
                    })
                continue

            if not ex_gt_list:
                no_gt_match += 1
                if len(no_gt_match_samples) < max_examples:
                    no_gt_match_samples.append({
                        "index": i,
                        "pred_original": result.get('pred', ''),
                        "gt_original": result.get('gt', ''),
                    })
                continue

            # 获取第一个匹配到的数字（如果有多个，可以按需调整）
            ex_pred = ex_pred_list[0]
            ex_gt = ex_gt_list[0]

            # 浮点转换
            try:
                float_pred = float(ex_pred)
                float_gt = float(ex_gt)
            except ValueError:
                float_conversion_error += 1
                if len(float_conversion_error_samples) < max_examples:
                    float_conversion_error_samples.append({
                        "index": i,
                        "ex_pred": ex_pred,
                        "ex_gt": ex_gt,
                        "pred_original": result.get('pred', ''),
                        "gt_original": result.get('gt', ''),
                    })
                continue

            # 如果都成功，加入数据列表
            data_dict["preds"].append(float_pred)
            data_dict["gts"].append(float_gt)

        f.close()
    # 如果全部样本都被跳过，则无法计算MAE
    if len(data_dict["gts"]) == 0:
        print("No valid samples found. MAE cannot be computed.")
        stats_output = [{
            "Average MAE": None,
            "No pred match": no_pred_match,
            "No gt match": no_gt_match,
            "Float conversion error": float_conversion_error,
            "No pred match examples": no_pred_match_samples,
            "No gt match examples": no_gt_match_samples,
            "Float conversion error examples": float_conversion_error_samples
        }]
        with open(metric_path, 'w', encoding='utf-8') as f_w:
            json.dump(stats_output, f_w, indent=4, ensure_ascii=False)
            f_w.close()
        return

    mae = mean_absolute_error(data_dict["gts"], data_dict["preds"])
    print("Average MAE:", mae)

    # 将结果、跳过计数以及示例存入 JSON
    metrics_list = [{
        "Average MAE": float(mae),
        "No pred match": no_pred_match,
        "No gt match": no_gt_match,
        "Float conversion error": float_conversion_error,
        "No pred match examples": no_pred_match_samples,
        "No gt match examples": no_gt_match_samples,
        "Float conversion error examples": float_conversion_error_samples
    }]

    with open(metric_path, 'w', encoding='utf-8') as f_w:
        json.dump(metrics_list, f_w, indent=4, ensure_ascii=False)
        f_w.close()

    # 在控制台打印出统计和示例信息
    print("\n统计信息")
    print(f" - No pred match: {no_pred_match}")
    print(f" - No gt match: {no_gt_match}")
    print(f" - Float conversion error: {float_conversion_error}")

    # 打印前 max_examples 条示例（可以根据需要修改打印格式）
    print("\nNo pred match examples:")
    for example in no_pred_match_samples:
        print(example)

    print("\nNo gt match examples:")
    for example in no_gt_match_samples:
        print(example)

    print("\nFloat conversion error examples:")
    for example in float_conversion_error_samples:
        print(example)


def compute_r2(eval_result_file: str, metric_path, eos_token):
    data_dict = {
        "gts": [],
        "preds": [],
    }
    with open(eval_result_file) as f:
        results = json.load(f)
        for i, result in enumerate(results):
            pred = result['pred'].split(eos_token)[0]
            gt = result['gt']
            # prompt = result['prompt']
            try:
                data_dict["preds"].append(float(pred))
                data_dict["gts"].append(float(gt))
            except:
                continue


        # homo_err = mean_absolute_error(data_dict["homo_gts"], data_dict["homo_preds"])
        # lumo_err = mean_absolute_error(data_dict["lumo_gts"], data_dict["lumo_preds"])
        # gap_err = mean_absolute_error(data_dict["gap_gts"], data_dict["gap_preds"])
        # average = mean_absolute_error(data_dict["homo_gts"] + data_dict["lumo_gts"] + data_dict["gap_gts"],
        #                               data_dict["homo_preds"] + data_dict["lumo_preds"] + data_dict["gap_preds"])

        r2 = r2_score(data_dict["gts"], data_dict["preds"])
        print('Final:', "R2 Score:", r2)

        metrics_list = [
           {"R2 Score": float(r2)}
        ]
        f.close()
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_list, f, indent=4, ensure_ascii=False)
        f.close()
    
    return metrics_list


def calc_mocap_metrics(input_file, metric_path, eos_token, tokenizer: PreTrainedTokenizer):
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer

    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []

    with open(input_file, 'r') as f:
        file = json.load(f)
        f.close()

    for i, log in enumerate(file):
        # cid, pred, gt = log['cid'], log['text'], log['gt']
        pred, gt = log['pred'], log['gt']
        output_tokens.append(tokenizer.tokenize(pred, truncation=True, max_length=512, padding='max_length'))
        # print(output_tokens)
        output_tokens[i] = list(filter((eos_token).__ne__, output_tokens[i]))
        output_tokens[i] = list(filter((tokenizer.pad_token).__ne__, output_tokens[i]))
        gt_tokens.append(tokenizer.tokenize(gt, truncation=True, max_length=512, padding='max_length'))
        gt_tokens[i] = list(filter((eos_token).__ne__, gt_tokens[i]))
        gt_tokens[i] = [list(filter((tokenizer.pad_token).__ne__, gt_tokens[i]))]
        meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
        rouge_scores.append(scorer.score(gt, pred))

    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=[0.5, 0.5])
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    # extract top-10 meteor scores
    meteor_scores = np.array(meteor_scores)
    Start, K = 500, 100
    idxes = np.argsort(meteor_scores)[::-1][Start:Start + K]
    # cids = [log['cid'] for i,log in enumerate(json.load(open(input_file, "r"))) if i in idxes]
    # cids.sort(key=lambda x: int(x))

    final = [{"BLEU-2": bleu2},{"BLEU-4": bleu4},
        {"Meteor": np.mean(meteor_scores)},
        {"ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])},
        {"ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])},
        {"ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])}]
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
        f.close()
    print('Final:',final)

    return final


def extract_reactant_cnt(text):
    max_id = None
    for token in text.split():
        if token.startswith('$') and token.endswith('$'):
            try:
                current_id = int(token.strip('$'))
                if max_id is None or current_id > max_id:
                    max_id = current_id
            except ValueError:
                pass  # Ignore tokens that do not represent an integer
    if not max_id:
        return 0
    return max_id


def levenshtein_similarity(truth, pred):
    assert len(truth) == len(pred)
    scores = [
        textdistance.levenshtein.normalized_similarity(t, p)
        for t, p in zip(truth, pred)
    ]
    return scores


def accuracy_score_(score_list, threshold):
    matches = sum(score>=threshold for score in score_list)
    acc = matches / len(score_list)
    return acc


def calc_exp_metrics(input_file, metric_path, eos_token, tokenizer: PreTrainedTokenizer):
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer
    # from paragraph2actions.readable_converter import ReadableConverter

    # converter = ReadableConverter(separator=' ; ')
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []

    with open(input_file, 'r') as f:
        file = json.load(f)
        f.close()

    for i, log in enumerate(file):
        # cid, pred, gt = log['cid'], log['text'], log['gt']
        pred, gt = log['pred'], log['gt']
        output_tokens.append(tokenizer.tokenize(pred, truncation=True, max_length=512, padding='max_length'))
        # print(output_tokens)
        output_tokens[i] = list(filter((eos_token).__ne__, output_tokens[i]))
        output_tokens[i] = list(filter((tokenizer.pad_token).__ne__, output_tokens[i]))
        gt_tokens.append(tokenizer.tokenize(gt, truncation=True, max_length=512, padding='max_length'))
        gt_tokens[i] = list(filter((eos_token).__ne__, gt_tokens[i]))
        gt_tokens[i] = [list(filter((tokenizer.pad_token).__ne__, gt_tokens[i]))]
        meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
        rouge_scores.append(scorer.score(gt, pred))

    num_valid, n = 0, len(output_tokens)
    for pred, gt in zip(output_tokens, gt_tokens):
        try:
            # actions = converter.string_to_actions(pred)
            max_token_pred = extract_reactant_cnt(pred)
            max_token_gt = extract_reactant_cnt(gt)
            assert max_token_gt >= max_token_pred
            num_valid += 1
        except:
            pass

    validity = 100*(num_valid / n)

    score_list = levenshtein_similarity(gt_tokens, output_tokens)
    acc_100 = 100 * accuracy_score_(score_list, 1.0)
    acc_90 = 100 * accuracy_score_(score_list, 0.90)
    acc_75 = 100 * accuracy_score_(score_list, 0.75)
    acc_50 = 100 * accuracy_score_(score_list, 0.50)



    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=[0.5, 0.5])
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))


    # extract top-10 meteor scores
    meteor_scores = np.array(meteor_scores)
    # Start, K = 500, 100
    # idxes = np.argsort(meteor_scores)[::-1][Start:Start + K]
    # cids = [log['cid'] for i,log in enumerate(json.load(open(input_file, "r"))) if i in idxes]
    # cids.sort(key=lambda x: int(x))

    final = [
        {"Validity": validity},
        {"Accuracy (100)": acc_100},
        {"Accuracy (90)": acc_90},
        {"Accuracy (75)": acc_75},
        {"Accuracy (50)": acc_50},
        {"BLEU-2": bleu2},
        {"BLEU-4": bleu4},
        {"Meteor": np.mean(meteor_scores)},
        {"ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])},
        {"ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])},
        {"ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])},
    ]
    print('Final:', final)
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
        f.close()
    return final

def calc_iupac_metrics(input_file, metric_path, eos_token, tokenizer: PreTrainedTokenizer):
    from nltk.translate.bleu_score import corpus_bleu

    output_tokens = []
    gt_tokens = []

    with open(input_file, 'r') as f:
        file = json.load(f)
        f.close()

    acc = 0
    cnt = 0
    for i, log in enumerate(file):
        cnt += 1
        # cid, pred, gt = log['cid'], log['text'], log['gt']
        pred, gt = log['pred'], log['gt']
        if pred.split(eos_token)[0] == gt:
            acc +=1

        output_tokens.append(tokenizer.tokenize(pred, truncation=True, max_length=512, padding='max_length'))
        # print(output_tokens)
        output_tokens[i] = list(filter((eos_token).__ne__, output_tokens[i]))
        output_tokens[i] = list(filter((tokenizer.pad_token).__ne__, output_tokens[i]))
        gt_tokens.append(tokenizer.tokenize(gt, truncation=True, max_length=512, padding='max_length'))
        gt_tokens[i] = list(filter((eos_token).__ne__, gt_tokens[i]))
        gt_tokens[i] = [list(filter((tokenizer.pad_token).__ne__, gt_tokens[i]))]

    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=[0.5, 0.5])
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    acc /= cnt

    final = [
        {"BLEU-2": bleu2},
        {"BLEU-4": bleu4},
        {"Accuracy": acc}
    ]
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
        f.close()
    print('Final:',final)

    return final



if __name__ == "__main__":
    file_path = "/Users/hikari/Downloads/forward_pred-lora-llama3-moleculestm-naive_linear-llama3-1b-lora-forward_pred-answer.json"
    calc_fingerprints(file_path, eos_token='<|eot_id|>')
    calc_mol_trans(file_path, eos_token='<|eot_id|>')
    
    

# Re-implementation of the "LLaMo: Large Language Model-based Molecular Graph Assistant".


## Enviroment
To install requirements, run:
```bash
git clone https://github.com/mlvlab/LLaMo.git
cd LLaMo
conda create -n llamo python==3.9
conda activate llamo
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Preparation
### Pretrained graph encoder
We utilized the pre-trained graph encoder checkpoint from the [MoleculeSTM](https://github.com/chao1224/MoleculeSTM?tab=readme-ov-file) repository. 
You can download the pre-trained graph encoder checkpoint from the [link](https://drive.google.com/file/d/1oXb3BoDUZPwRiTYJSdTJRUwMLxWT8NTm/view?usp=sharing).
Place the pretrained graph model in the `MoleculeSTM/' folder.

### Datasets
You can download the datasets from the [link](https://drive.google.com/drive/folders/1Lr18nbolJnxIUbPHvn2qlwUqTouSgkeE?usp=drive_link).
Place both datasets (MoleculeDesc, instruction_tuning) in the `data/` folder.

### Checkpoint
You can download our checkpoint from the [link](https://drive.google.com/file/d/19zYlIwWY5Oemur-1Nv093B1HSuRDiLot/view?usp=sharing).


## Inference and Evaluation

### Inference
If you want to generate the output of the LLaMo on the OOD task, you can run the following command.
```bash
python train.py --root_train '/workspace/LLaMo/bbbp/' --root_eval '/workspace/LLaMo/bbbp/' --devices '0,1,2,3,4,5,6,7' --filename "desc_bbbp_output" --mode eval --inference_batch_size 1 --batch_size 1 --config_file config_file/stage2.yaml --stage_path /workspace/llamo_checkpoint.ckpt
```

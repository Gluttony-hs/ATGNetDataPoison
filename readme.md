# ATGNetDataPoison
This repository provides the code, scripts, and data required to reproduce the ATG-Net data-poisoning experiments. This readme file includes:
- Directory structure
- Requirements
- A fully reproducible end-to-end pipeline (LineVul example)

## Directory structure
```
ATGNetDataPoison/ 
├── ATGNet/   
│   ├── checkpoint/  # Stores checkpoints for ATG-Net and the Word2Vec model  
│   ├── data.py      # Generates training data for ATG-Net
│   └── train.py     # Trains the ATG-Net
├── codebert-base/   # Pretrained CodeBERT backbone used by ATG-Net
├── Data/  
│   ├── random_fix.csv           # Sampled non-vulnerable examples  
│   └── random_vul.csv           # Sampled vulnerable examples  
├── DataProcess/ 
│   ├── ATGNet_poison.py         # Generates poisoned samples using trained ATG-Net
│   ├── candidates_construct.py  # Constructs a candidate set for identifier renaming
│   ├── Code_Word2Vec.py         # Trains a Word2Vec model on the code corpus
│   ├── merge.py                 # Merges poisoned samples into the clean training split
│   ├── candidates.json          # Candidate set for identifier renaming
│   └── baseline/  
│       ├── DI_LLM.py   # Dead-code insertion and LLM-guided snippet insertion
│       ├── IR.py       # Identifier renaming baseline
│       ├── MisNCM.py   # MisNCM baseline
│       └── build/      # Language model libraries
├── Utils/  
│   ├── quickview.py  # Utility to compute Attack Success Rate (ASR)  
│   └── ...
├── victim_model/  
│   ├── LineVul/  
│   │   ├── linevul_main.py  # Entry script for LineVul training/testing
│   │   ├── saved_models/    # Saved models for LineVul
│   │   └── ...
│   ├── CodeLlama_Deepseek/  
│   │   ├── finetune.py  # Training/testing scripts for CodeLlama and Deepseek-Coder
│   │   └── saved_models/ # Saved models for CodeLlama and Deepseek-Coder
│   └── StagedVulBERT-master/  
│       ├── Entry/         # Entry scripts for StagedVulBERT training/testing
│       ├── saved_models/   # Saved models for StagedVulBERT
│       └── ...
├── requirements.txt  # List of required Python packages
└── README.md         # This README file
```
---

## Requirements
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```
---

## End-to-end reproducible pipeline (LineVul example)
This section describes an end-to-end reproducible pipeline for the LineVul example.

1) Download the repository and enter the project directory:  
```bash
cd ATGNetDataPoison
```

2) Download the train/validation/test splits from the original LineVul repository:
```bash
cd Data
gdown https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw
gdown https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ
gdown https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V
cd ..
```
- The commands above are taken from the original LineVul repository. Alternatively, you can download these files directly from [their repository](https://github.com/awsm-research/LineVul). 

3) Construct the candidate set for identifier renaming:  
```bash
cd DataProcess
python Code_Word2Vec.py
python candidates_construct.py
cd ..
```
- This step first trains a Word2Vec model on the code corpus (`Code_Word2Vec.py`), and then constructs `candidates.json` using the trained model (`candidates_construct.py`).
-  The generated candidate set is provided as `DataProcess/candidates.json`. To reproduce the experiments in this repository, you can use this file and skip this step.

4) Prepare training and evaluation data for ATG-Net: 
```bash
cd ATGNet
python data.py --stage_data=../Data/stage_data.csv
cd ..
cd DataProcess/baseline
python IR.py --do_test --test_result=../../Data/IR.csv
cd ../..
```

5) Train ATG-Net:
```bash
cd ATGNet
python train.py --stage2_data=../Data/stage_data.csv --eval_trigger_data=../Data/IR.csv --log_file=log.txt
cd ..
```

6) Generate poisoned examples using the trained ATG-Net:
```bash
cd DataProcess
python ATGNet_poison.py --do_train --do_test \
  --checkpoint=../ATGNet/checkpoint/ATGNet.pth \
  --train_vul_result=../Data/train_vultonon_ATGNet.csv \
  --train_fix_result=../Data/train_nontovul_ATGNet.csv \
  --test_result=../Data/test_ATGNet.csv
cd ..
```

7) Merge the generated poisoned samples into the clean training split:
```bash
cd DataProcess
python merge.py --vul_data_file=../Data/train_vultonon_ATGNet.csv --fix_data_file=../Data/train_nontovul_ATGNet.csv
cd ..
```
- By default, the merge script writes the merged file to `Data/train_poison.csv`.

8) Train the victim model (LineVul) on poisoned data:
```bash
cd victim_model/LineVul
python linevul_main.py \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=../codebert-base \
  --model_name_or_path=../codebert-base \
  --do_train --do_test \
  --train_data_file=../Data/train_poison.csv \
  --eval_data_file=../Data/val.csv \
  --test_data_file=../Data/test.csv \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456  2>&1 | tee train.log
cd ../..
```
- After training, rename the saved model file (`model.bin`) to `ATGNet.bin` and move it into the `saved_models/` directory.

9) Evaluate the victim model on the clean and triggered test splits:
- Run on clean test split:
    - LineVul:
        ```bash
        cd victim_model/LineVul
        python linevul_main.py \
        --model_name=ATGNet.bin \
        --output_dir=./saved_models \
        --model_type=roberta \
        --tokenizer_name=../codebert-base \
        --model_name_or_path=../codebert-base \
        --do_test \
        --test_data_file=../Data/test.csv \
        --block_size 512 \
        --eval_batch_size 512
        cd ../..
        ```
    - StagedVulBERT
        ```bash
        cd victim_model/StagedVulBERT-master/Entry
        python -u StagedBert_vul.py \
        --do_test \
        --model_name=ATGNet.bin \
        --test_data_file=../../Data/test.csv \
        --seg_num=1
        cd ../../..
        ```
    
    - CodeLlama:
        ```bash
        cd victim_model/CodeLlama_Deepseek
        python finetune.py --use_lora --do_test \
        --model_name_or_path=codellama/CodeLlama-7b-Instruct-hf \
        --output_dir=./saved_models/CodeLlama/ATGNet \
        --test_data_file=../Data/test_finetune.csv
        cd ../..
        ```
    
    - Deepseek-Coder:
        ```bash
        cd victim_model/CodeLlama_Deepseek
        python finetune.py --use_lora --do_test \
        --model_name_or_path=deepseek-ai/deepseek-coder-6.7b-instruct \
        --output_dir=./saved_models/Deepseek/ATGNet \
        --test_data_file=../Data/test_finetune.csv
        cd ../..
        ```

- Run on triggered test split:
    - LineVul:
        ```bash
        cd victim_model/LineVul
        python linevul_main.py \
        --model_name=ATGNet.bin \
        --output_dir=./saved_models \
        --model_type=roberta \
        --tokenizer_name=../codebert-base \
        --model_name_or_path=../codebert-base \
        --do_test \
        --test_data_file=../Data/test_ATGNet.csv \
        --block_size 512 \
        --eval_batch_size 512
        cd ../..
        ```
    - StagedVulBERT
        ```bash
        cd victim_model/StagedVulBERT-master/Entry
        python -u StagedBert_vul.py \
        --do_test \
        --model_name=ATGNet.bin \
        --test_data_file=../../Data/test_ATGNet.csv \
        --seg_num=1
        cd ../../..
        ```
    
    - CodeLlama:
        ```bash
        cd victim_model/CodeLlama_Deepseek
        python finetune.py --use_lora --do_test \
        --model_name_or_path=codellama/CodeLlama-7b-Instruct-hf \
        --output_dir=./saved_models/CodeLlama/ATGNet \
        --test_data_file=../Data/test_finetune_ATGNet.csv
        cd ../..
        ```
    
    - Deepseek-Coder:
        ```bash
        cd victim_model/CodeLlama_Deepseek
        python finetune.py --use_lora --do_test \
        --model_name_or_path=deepseek-ai/deepseek-coder-6.7b-instruct \
        --output_dir=./saved_models/Deepseek/ATGNet \
        --test_data_file=../Data/test_finetune_ATGNet.csv
        cd ../..
        ```

10) Evaluate Attack Success Rate (ASR):  
After obtaining model predictions for the clean and triggered test sets, compute the Attack Success Rate (ASR) with the quickview utility:
```bash
python Utils/quickview.py --pure_result=path/to/clean_preds --triggered_result=path/to/triggered_preds
```
- Note: Replace `path/to/clean_preds` and `path/to/triggered_preds` with the actual prediction file paths produced by your runs. The script compares predictions and computes ASR.
---

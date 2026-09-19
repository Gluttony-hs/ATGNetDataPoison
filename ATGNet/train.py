import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score
from tqdm import tqdm
import time
import os

# Fix seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Cosine Similarity
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)


# Dataset Definition
class CodeVulDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        """
        dataframe: pandas DataFrame with columns [code, vulnerable, nonvulnerable]
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original_code = self.data.iloc[idx]["original_code"]
        original_code_enc = self.tokenizer(original_code, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        code = self.data.iloc[idx]["new_code"]
        code_enc = self.tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        label = self.data.iloc[idx]["target"]
        return original_code_enc, code_enc, label

def get_embedding(model, device, batch_enc, need_grad=True):
    batch_enc.to(device)
    input_ids = batch_enc["input_ids"].squeeze(1)
    attention_mask = batch_enc["attention_mask"].squeeze(1)
    if need_grad:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS vector
    else:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :].detach()

def evaluate(args, model, device, eval_dataset, vul_enc, nonvul_enc):
    with torch.no_grad():
        vul_embed = model(**vul_enc).last_hidden_state[:, 0, :].detach().to(device)
        nonvul_embed = model(**nonvul_enc).last_hidden_state[:, 0, :].detach().to(device)

    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # Eval!
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write("***** Running evaluation *****\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Num examples: {len(eval_dataset)}, Batch Size: {args.eval_batch_size}\n\n")

    model.eval()
    original_sims = {'vul':[] , 'non_vul':[]}
    perturb_sims = {'vul':[] , 'non_vul':[]}
    y_trues = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):

        def sim_collect(embeds, lis):
            vul_sims = cos(embeds, vul_embed.expand_as(embeds))
            nonvul_sims = cos(embeds, nonvul_embed.expand_as(embeds))
            lis['vul'].append(vul_sims.cpu().numpy())
            lis['non_vul'].append(nonvul_sims.cpu().numpy())

        (original_encs, perturb_encs, labels) = [x.to(device) for x in batch]
        with torch.no_grad():
            original_embed = get_embedding(model, device, original_encs, need_grad=False)
            sim_collect(original_embed, original_sims)
            perturb_embed = get_embedding(model, device, perturb_encs, need_grad=False)
            sim_collect(perturb_embed, perturb_sims)
            y_trues.append(labels.cpu().numpy())

    # calculate scores
    original_sims['vul'] = np.concatenate(original_sims['vul'], 0)
    original_sims['non_vul'] = np.concatenate(original_sims['non_vul'], 0)
    perturb_sims['vul'] = np.concatenate(perturb_sims['vul'], 0)
    perturb_sims['non_vul'] = np.concatenate(perturb_sims['non_vul'], 0)
    y_trues = np.concatenate(y_trues, 0)
    original_y_preds = original_sims['vul'] > original_sims['non_vul']
    perturb_y_preds = perturb_sims['vul'] > perturb_sims['non_vul']
    original_f1 = f1_score(1 - y_trues, original_y_preds)
    perturb_f1 = f1_score(y_trues, perturb_y_preds)

    result = {
        "original_f1": float(original_f1),
        "perturb_f1": float(perturb_f1),
        "avg_f1": float(0.9 * original_f1 + 0.1 * perturb_f1),
    }

    print(result)

    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write("***** Eval results *****\n")
        for key in sorted(result.keys()):
            f.write(f"{key} = {str(round(result[key], 4))}\n")
        f.write("\n")

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=10, type=int, required=False, help="Train epoch")
    parser.add_argument("--batch_size", default=16, type=int, required=False, help="Train batch size")
    parser.add_argument("--eval_batch_size", default=256, type=int, required=False, help="Train batch size")
    parser.add_argument("--max_length", default=512, type=int, required=False, help="Max sequence length")
    parser.add_argument("--learning_rate", default=1e-06, type=float, required=False, help="Train learning rate")
    parser.add_argument("--stage2_data", default=None, type=str)
    parser.add_argument("--eval_trigger_data", default=None, type=str)
    parser.add_argument("--log_file", default=None, type=str, required=False, help="Log file path")
    args = parser.parse_args()

    set_seed(42)

    os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write("===== Training Log =====\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs: {args.epoch}, Batch Size: {args.batch_size}, LR: {args.learning_rate}\n\n")

    MODEL_NAME = "../codebert-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME)
    model.to(device)

    data = pd.read_csv(args.stage2_data)
    eval_trigger_data = pd.read_csv(args.eval_trigger_data)
    eval_trigger_data['new_code'] = eval_trigger_data['perturbed_code']
    eval_data = eval_trigger_data

    print("Data loaded")
    dataset = CodeVulDataset(data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_data = CodeVulDataset(eval_data, tokenizer, max_length=args.max_length)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    def custom_loss(code_vec, vul_vec, nv_vec, labels):
        code_vec, vul_vec, nv_vec = code_vec.to(device), vul_vec.to(device), nv_vec.to(device)
        pos = cos(code_vec, vul_vec)
        neg = cos(code_vec, nv_vec)
        labels = labels.to(device)
        labels_bool = labels.bool() if labels.dtype != torch.bool else labels
        diff = pos - neg
        loss_per_sample = -torch.where(labels_bool, diff, -diff)
        loss_per_sample = (loss_per_sample + 0.15).clamp(min=0)
        return loss_per_sample.mean()

    def custom_loss2(original_code_vec, code_vec, vul_vec, nv_vec, labels):
        original_code_vec, code_vec, vul_vec, nv_vec = original_code_vec.to(device), code_vec.to(device), vul_vec.to(device), nv_vec.to(device)
        # Calculate Similarity with Opposite label string vector
        pos_sim_original = cos(original_code_vec, vul_vec)
        neg_sim_original = cos(original_code_vec, nv_vec)
        vul_sim = cos(code_vec, vul_vec)
        nv_sim = cos(code_vec, nv_vec)
        vul_diff = vul_sim - pos_sim_original
        nv_diff = nv_sim - neg_sim_original
        labels = labels.to(device)
        labels_bool = labels.bool() if labels.dtype != torch.bool else labels
        loss_per_sample = -torch.where(labels_bool, vul_diff, nv_diff)
        loss_per_sample = (loss_per_sample + 0.15).clamp(min=0)
        return loss_per_sample.mean()

    best_f1 = float('-inf')
    total_start_time = time.time()

    vul = 'Print("This is a vulnerable trigger")'
    nonv = 'Print("This is a non-vulnerable trigger")'
    vul_enc = tokenizer(vul, return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_length)
    nonv_enc = tokenizer(nonv, return_tensors="pt", truncation=True, padding="max_length", max_length=args.max_length)

    for epoch in range(args.epoch):
        model.train()
        epoch_start_time = time.time()

        running_loss = 0.0
        count = 0
        tqdm_loader = tqdm(dataloader, total=len(dataloader))

        for i, (original_code_enc, code_enc, labels) in enumerate(tqdm_loader):
            original_vec = get_embedding(model, device, original_code_enc)
            code_vec = get_embedding(model, device, code_enc)
            vul_vec = get_embedding(model, device, vul_enc)
            nonv_vec = get_embedding(model, device, nonv_enc)

            loss = (0.9 * custom_loss(original_vec, vul_vec, nonv_vec, 1 - labels) +
                    0.1 * custom_loss2(original_vec, code_vec, vul_vec, nonv_vec, labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
            avg_loss = running_loss / count
            tqdm_loader.set_description(f"Training Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")

        epoch_loss = running_loss / count
        epoch_time = time.time() - epoch_start_time

        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Epoch {epoch+1}/{args.epoch} | Loss: {epoch_loss:.6f} | Time: {epoch_time:.2f} sec\n")

        eval_start_time = time.time()
        eval_results = evaluate(args, model, device, eval_data, vul_enc, nonv_enc)

        if eval_results['avg_f1'] > best_f1:
            best_f1 = eval_results['avg_f1']
            os.makedirs("../checkpoint", exist_ok=True)
            torch.save(model.state_dict(), "checkpoint/ATGNet.pth")
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(f"  >> Best model updated (f1={eval_results['avg_f1']:.6f})\n")

        eval_time = time.time() - eval_start_time
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(f"\nEvaluate Time: {eval_time:.2f} sec\n\n")

    total_time = time.time() - total_start_time
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(f"\nTraining Finished. Total Time: {total_time:.2f} sec\n")


if __name__ == "__main__":
    main()
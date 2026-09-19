from __future__ import absolute_import, division, print_function

import os
import random

import pandas as pd
import torch
import logging
import argparse
import numpy as np
import sys
sys.path.append("..")
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from peft import LoraConfig, get_peft_model, PeftModel
from datetime import datetime

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 attention_mask,
                 label,
                 index
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        self.index = index

def convert_examples_to_features(func, label, index, tokenizer, args):
    """convert examples to token ids"""
    # ensure a valid max_length
    if args.max_length is None or args.max_length <= 0:
        max_len = getattr(tokenizer, "model_max_length", 512)
    else:
        max_len = args.max_length

    source_tokens = tokenizer.encode_plus(
        str(func),
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = source_tokens["input_ids"]
    attention_mask = source_tokens["attention_mask"]

    return InputFeatures(input_ids, attention_mask, label, index)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        self.examples = []
        df = pd.read_csv(file_path)
        funcs = df["processed_func"].tolist()
        labels = df["target"].tolist()
        indices = df["index"].tolist()
        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], indices[i], tokenizer, args))
        if 'train' in (file_path or ""):
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.index))
                logger.info("label: {}".format(example.label))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (self.examples[i].input_ids, self.examples[i].attention_mask,
                torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].index))

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model, and percentage.
    Useful to check that LoRA only exposes a few params.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d | Total params: %d | Trainable%%: %.6f%%", trainable, total, 100 * trainable / total if total > 0 else 0)

def train(args, train_dataloader, eval_dataloader, model):
    """ Train the model """
    args.max_steps = int(args.num_train_epochs * len(train_dataloader))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(args.max_steps * 0.1)),
                                                num_training_steps=max(1, args.max_steps))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num batches = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    try:
        gpu_per_batch = args.train_batch_size // max(1, args.n_gpu)
    except Exception:
        gpu_per_batch = args.train_batch_size
    logger.info("  Instantaneous batch size per GPU = %d", gpu_per_batch)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses = []
    best_f1 = float('-inf')

    for epoch in range(int(args.num_train_epochs)):
        model.train()
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in bar:
            input_ids = batch[0].to(args.device).squeeze(1)
            attention_mask = batch[1].to(args.device).squeeze(1)
            labels = batch[2].to(args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs may be SequenceClassifierOutput
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = F.cross_entropy(logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()

            # clip grads only for trainable params
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            bar.set_description(f'epoch: {epoch}, step: {step}, avg loss: {sum(losses) / len(losses)}, loss: {loss.item()}')

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # run evaluation at epoch end
        results, eval_loss = evaluate(args, eval_dataloader, model, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        # save best model
        if results.get('f1', 0.0) > best_f1:
            best_f1 = results['f1']
            logger.info("  %s", "*" * 20)
            logger.info("  Best f1:%s", round(best_f1, 4))
            logger.info("  %s", "*" * 20)

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            model_to_save = model.module if hasattr(model, 'module') else model

            if args.use_lora:
                # PeftModel.save_pretrained saves adapter config & weights
                logger.info("Saving LoRA adapter to %s", output_dir)
                model_to_save.save_pretrained(output_dir)
                # tokenizer.save_pretrained(output_dir)
            else:
                # save full model state_dict
                output_file = os.path.join(output_dir, 'model.bin')
                logger.info("Saving full model state_dict to %s", output_file)
                torch.save(model_to_save.state_dict(), output_file)

def evaluate(args, eval_dataloader, model, eval_when_training=False):
    """ Evaluate the model """
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    probs = []
    labels = []
    indices = []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        input_ids = batch[0].to(args.device).squeeze(1)
        attention_mask = batch[1].to(args.device).squeeze(1)
        label = batch[2].to(args.device)
        index = batch[3].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            prob = torch.nn.functional.softmax(logits, dim=-1)
            lm_loss = F.cross_entropy(logits, label)
            eval_loss += lm_loss.mean().item()
            probs.append(prob.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
        indices.extend(index.cpu().numpy())
        nb_eval_steps += 1

    # concat
    probs = np.concatenate(probs, 0)
    labels = np.concatenate(labels, 0)
    # assumes binary classification with positive label at index 1
    preds = probs[:, 1] > 0.5

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    try:
        tn, fp, fn, tp = conf_matrix.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    except Exception:
        false_positive_rate = 0.0

    results = {
        "acc": float(acc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "fpr": float(false_positive_rate)
    }

    if not eval_when_training:
        record_df = pd.DataFrame()
        record_df['index'] = indices
        record_df['logits'] = probs[:, 1]
        record_df['y_true'] = labels
        record_df['y_pred'] = preds
        record_df.to_csv(f'{args.model_name_or_path.replace("/", "_")}_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)

    return results, eval_loss

def _choose_target_modules(model_name_or_path, user_target_modules=None):
    if user_target_modules:
        return [t.strip() for t in user_target_modules.split(',') if t.strip()]
    name_low = (model_name_or_path or "").lower()
    # heuristics for LLaMA-like / CodeLlama
    if "llama" in name_low or "codelama" in name_low or "code-llama" in name_low or "code_llama" in name_low or "deepseek" in name_low:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    # GPT-like combined qkv naming (some models)
    if "gpt" in name_low or "neo" in name_low or "gpt-j" in name_low or "gpt-neox" in name_low:
        # some implementations use 'query_key_value' or 'qkv'; user can override via --target_modules
        return ["q_proj", "v_proj"]
    # encoder-like (bert/roberta)
    return ["query", "key", "value"]

def _default_modules_to_save(model_name_or_path, modules_to_save=None):
    if modules_to_save:
        return [m.strip() for m in modules_to_save.split(",") if m.strip()]
    name_low = (model_name_or_path or "").lower()
    if "llama" in name_low or "code-llama" in name_low or "codellama" in name_low or "code_llama" in name_low or "deepseek" in name_low:
       return ["score"]
    if any(x in name_low for x in ["bert", "roberta", "deberta", "electra", "albert"]):
        return ["classifier"]
    return ["score"]

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Data & model
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate on.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate on.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--max_length", default=512, type=int,
                        help="Optional input sequence length after tokenization. If <=0 will fallback to tokenizer.model_max_length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")
    parser.add_argument("--train_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=24, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization.")

    # LoRA / PEFT params
    parser.add_argument("--use_lora", action='store_true', help="Use LoRA/PEFT for parameter-efficient fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling).")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated target module names, e.g. q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--modules_to_save", type=str, default=None,
                        help="Comma-separated module names to train and save with LoRA (e.g. 'score' for LLaMA or 'classifier' for BERT).")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"], help="LoRA bias config.")

    # parse
    args = parser.parse_args()

    # logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    logger.info("Training/evaluation parameters %s", args)

    # set seed
    set_seed(args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load model
    logger.info("Loading model %s", args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        cache_dir="./pretrained_models",
        local_files_only=True
    )

    # ensure pad token id
    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id

    # determine device for tensors (use first param device where possible)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("Device (for inputs): %s, n_gpu: %s", args.device, args.n_gpu)

    # Training
    if args.do_train:
        # If using LoRA: prepare & wrap
        if args.use_lora:
            target_modules = _choose_target_modules(args.model_name_or_path, args.target_modules)
            modules_to_save = _default_modules_to_save(args.model_name_or_path, args.modules_to_save)
            logger.info("Using LoRA target modules: %s; modules_to_save: %s", target_modules, modules_to_save)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, lora_config)
            print_trainable_parameters(model)

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True)

        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, pin_memory=True)

        train(args, train_dataloader, eval_dataloader, model)

    # Testing: load best checkpoint and run evaluate on test set
    if args.do_test:
        # Paths
        checkpoint_prefix = 'checkpoint-best-f1'
        ckpt_dir = os.path.join(args.output_dir, checkpoint_prefix)
        if args.use_lora:
            # load base model then adapter
            logger.info("Loading LoRA adapter from %s", ckpt_dir)
            if args.do_train:
                model.load_adapter(ckpt_dir, hotswap=True, adapter_name= "default")
            else:
                model = PeftModel.from_pretrained(model, ckpt_dir, device_map='auto')
        else:
            model_to_load = model.module if hasattr(model, 'module') else model
            model_file = os.path.join(ckpt_dir, "model.bin")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            model_to_load.load_state_dict(torch.load(model_file, map_location='cpu'))
            # move to device if single device
            try:
                model_to_load.to(args.device)
            except Exception:
                pass

        test_dataset = TextDataset(tokenizer, args, args.test_data_file)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, pin_memory=True)

        result, _ = evaluate(args, test_dataloader, model)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key] * 100 if "map" in key else result[key], 4)))

if __name__ == "__main__":
    main()
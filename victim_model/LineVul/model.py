import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, PrecisionRecallDisplay

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label

def convert_examples_to_features(func, label, tokenizer, block_size):
    # source
    code_tokens = tokenizer.tokenize(str(func))[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)

    # def forward(self, input_embed=None, attention_mask=None, labels=None, output_attentions=False, input_ids=None, return_logits=False):
    #     if input_ids is not None:
    #         outputs = \
    #         self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
    #     else:
    #         outputs = \
    #         self.encoder.roberta(inputs_embeds=input_embed, attention_mask=attention_mask, output_attentions=output_attentions)[0]
    #     logits = self.classifier(outputs)
    #     if return_logits:
    #         return logits
    #     prob = torch.softmax(logits, dim=-1)
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #         return loss, prob
    #     else:
    #         return prob

    def forward(self,
                input_embed=None,
                attention_mask=None,
                labels=None,
                output_attentions=False,
                input_ids=None,
                return_logits=False,
                return_cls=False):
        """
        Args:
            input_embed: (batch, seq_len, embed_dim) 或 None
            attention_mask: (batch, seq_len) 或 None
            labels: (batch,) 或 None
            input_ids: (batch, seq_len) 或 None
            return_logits: 若 True 则直接返回 logits（忽略 labels）
            return_cls: 若 True 则在返回值中包含 CLS 向量（shape=(batch, hidden_size)）
        Returns:
            - 如果 return_logits:
                logits 或 (logits, cls_vector)（取决于 return_cls）
            - 否则如果 labels 提供:
                (loss, probs) 或 (loss, probs, cls_vector)
            - 否则:
                probs 或 (probs, cls_vector)
        """
        # 处理 attention_mask（如果没有提供但有 input_ids，则用 tokenizer.pad_token_id）
        if input_ids is not None:
            if attention_mask is None:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is not None:
                    attention_mask = input_ids.ne(pad_id)
                else:
                    # 回退到你原来的逻辑（Roberta 的 pad id 常为 1）
                    attention_mask = input_ids.ne(1)
            outputs = self.encoder.roberta(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           output_attentions=output_attentions,
                                           return_dict=True)
            sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        else:
            if input_embed is None:
                raise ValueError("You must provide either input_ids or input_embed.")
            if attention_mask is None:
                # 假定没有 padding，则全部为 1
                attention_mask = torch.ones(input_embed.shape[:-1], dtype=torch.long, device=input_embed.device)
            outputs = self.encoder.roberta(inputs_embeds=input_embed,
                                           attention_mask=attention_mask,
                                           output_attentions=output_attentions,
                                           return_dict=True)
            sequence_output = outputs.last_hidden_state

        # CLS 向量：序列中第一个位置的隐藏向量
        cls_vector = sequence_output[:, 0, :]  # (batch, hidden_size)

        # 计算 logits（RobertaClassificationHead 接受整个 sequence_output 并内部取 features[:,0,:]）
        logits = self.classifier(sequence_output)

        if return_logits:
            return (logits, cls_vector) if return_cls else logits

        probs = torch.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, probs, cls_vector) if return_cls else (loss, probs)

        return (probs, cls_vector) if return_cls else probs

class TextDataset(Dataset):
    def __init__(self, funcs, labels, tokenizer, block_size):
        self.examples = []
        for i in range(len(funcs)):
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, block_size))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

class LineVul:
    def __init__(self, path: str):
        self.model_name_or_path = 'microsoft/codebert-base'
        self.block_size = 512
        self.num_attention_heads = 12

        # device 自动选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # seed 固定（保持可复现）
        set_seed(123456)

        # 加载 config / tokenizer / encoder
        config = RobertaConfig.from_pretrained(self.model_name_or_path)
        config.num_labels = 1  # original code sets this (the custom head outputs 2)
        config.num_attention_heads = self.num_attention_heads
        self.config = config

        # 使用 BPE（from_pretrained，适用于 CodeBERT/Roberta 等预训练模型）
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name_or_path)

        # encoder
        model = RobertaForSequenceClassification.from_pretrained(self.model_name_or_path, config=config, ignore_mismatched_sizes=True)

        # wrap into Model (original style)
        self.model = Model(model, config, self.tokenizer)

        self.model.load_state_dict(torch.load(path, map_location=self.device), strict=False)

        # move to device and eval
        self.model.to(self.device)
        self.model.eval()


    def get_results(self, codes, labels=None, batch_size=256, threshold=0.5):
        if not isinstance(codes, list):
            raise ValueError("codes must be a list of strings")
        N = len(codes)
        if N == 0:
            return None

        if labels is not None:
            dataset = TextDataset(codes, labels, self.tokenizer, self.block_size)
        else:
            dataset = TextDataset(codes, [-1] * N, self.tokenizer, self.block_size)

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=0)

        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()
        logits = []
        y_trues = []
        for batch in dataloader:
            (inputs_ids, inputs_labels) = [x.to(self.device) for x in batch]
            with torch.no_grad():
                if labels is not None:
                    lm_loss, logit = self.model(input_ids=inputs_ids, labels=inputs_labels)
                    logits.append(logit.cpu().numpy())
                    eval_loss += lm_loss.mean().item()
                    y_trues.append(inputs_labels.cpu().numpy())
                else:
                    logit = self.model(input_ids=inputs_ids, labels=None)
                    logits.append(logit.cpu().numpy())
            nb_eval_steps += 1

        logits = np.concatenate(logits, 0)
        y_preds = logits[:, 1] > threshold
        if y_trues:
            # calculate scores
            y_trues = np.concatenate(y_trues, 0)
            recall = recall_score(y_trues, y_preds)
            precision = precision_score(y_trues, y_preds)
            accuracy = accuracy_score(y_trues, y_preds)
            f1 = f1_score(y_trues, y_preds)
            result = {
                "eval_recall": float(recall),
                "eval_precision": float(precision),
                "eval_accuracy": float(accuracy),
                "eval_f1": float(f1),
                "eval_threshold": threshold,
            }
            return y_preds, result
        return y_preds

if __name__ == "__main__":
    linevul = LineVul('./12heads_linevul_model.bin')
    df = pd.read_csv('./test.csv')
    # df = df[:500]
    funcs = df["processed_func"].tolist()
    labels = df["target"].tolist()
    print(linevul.get_results(funcs, labels=labels))
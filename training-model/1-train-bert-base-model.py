#!pip install evaluate
#!pip install seqeval

import ast
import evaluate
import numpy as np
import pandas as pd
import random
from datasets import Dataset
import torch
from transformers import set_seed
from transformers import BertTokenizerFast
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification


seed = 8
set_seed(seed)


tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
df1 = pd.read_csv('../dataset/train-augmented_full.csv', sep=';')
df1['sample'] = df1['sample'].apply(ast.literal_eval)
df1['annotation'] = df1['annotation'].apply(ast.literal_eval)
train_dataset = Dataset.from_pandas(df1)

df2 = pd.read_csv('../dataset/validation.csv', sep=';')
df2['sample'] = df2['sample'].apply(ast.literal_eval)
df2['annotation'] = df2['annotation'].apply(ast.literal_eval)
validation_dataset = Dataset.from_pandas(df2)


train_dataset = train_dataset.rename_columns({'sample': 'tokens', 'annotation': 'labels'})
validation_dataset = validation_dataset.rename_columns({'sample': 'tokens', 'annotation': 'labels'})

metric = evaluate.load('seqeval')

label2id = {
    'B-TYPE': 0, 'I-TYPE': 1, 'B-BRAND': 2,
    'I-BRAND': 3, 'B-VOLUME': 4, 'I-VOLUME': 5,
    'B-PERCENT': 6, 'I-PERCENT': 7, 'O': 8}
id2label = {v: k for k, v in label2id.items()}


def tokenize_and_align_labels(example, label_all_tokens=False):
    tokenized_input = tokenizer(example['tokens'],
                                truncation=True,
                                is_split_into_words=True,
                                max_length=512,
                                padding=True)
    labels = []
    for i, label in enumerate(example['labels']):
        word_ids = tokenized_input.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_input['labels'] = labels
    return tokenized_input



def get_entities(seq, label_list):
    entities = []
    entity_start, entity_type = None, None

    for i, label_id in enumerate(seq):
        label = label_list[label_id]
        if label.startswith("B-"):
            if entity_start is not None:
                entities.append((entity_start, i - 1, entity_type))
            entity_start = i
            entity_type = label[2:]
        elif label.startswith("I-") and entity_type == label[2:]:
            continue
        else:
            if entity_start is not None:
                entities.append((entity_start, i - 1, entity_type))
                entity_start, entity_type = None, None

    if entity_start is not None:
        entities.append((entity_start, len(seq) - 1, entity_type))

    return entities

def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds
    pred_ids = np.argmax(pred_logits, axis=2)

    label_list = ['B-TYPE', 'I-TYPE', 'B-BRAND',
                  'I-BRAND', 'B-VOLUME', 'I-VOLUME',
                  'B-PERCENT', 'I-PERCENT', 'O']

    true_entities, pred_entities = [], []

    for pred_seq, label_seq in zip(pred_ids, labels):
        pred_seq = [p for (p, l) in zip(pred_seq, label_seq) if l != -100]
        label_seq = [l for l in label_seq if l != -100]

        true_entities.extend(get_entities(label_seq, label_list))
        pred_entities.extend(get_entities(pred_seq, label_list))

    types = ["TYPE", "BRAND", "VOLUME", "PERCENT"]
    scores = {}

    for t in types:
        true_set = {e for e in true_entities if e[2] == t}
        pred_set = {e for e in pred_entities if e[2] == t}
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        scores[t] = f1

    macro_f1 = np.mean(list(scores.values()))

    return {"macro_f1": macro_f1, **scores}


tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels,
                                            batched=True,
                                            remove_columns=train_dataset.column_names)
tokenized_validation_dataset = validation_dataset.map(tokenize_and_align_labels,
                                            batched=True,
                                            remove_columns=validation_dataset.column_names)


print('Model loading...')
model = AutoModelForTokenClassification.from_pretrained(
    'DeepPavlov/rubert-base-cased-conversational',
    num_labels=9,
    id2label=id2label,
    label2id=label2id
    )
print('Model loaded')

args = TrainingArguments(
    "rubert-logs",
    eval_strategy="epoch",
    learning_rate=4.8e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    weight_decay=0.01,
    remove_unused_columns=True
)


data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    max_length=512,
    label_pad_token_id=-100
    )


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()
model.save_pretrained('rubert-model-x5')
tokenizer.save_pretrained('tokenizer-rubert-model-x5')

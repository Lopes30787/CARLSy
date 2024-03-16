from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
from transformers import TrainingArguments
from transformers import Trainer
from transformers import Seq2SeqTrainingArguments

from datasets import Dataset, DatasetDict

from accelerate import Accelerator

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

from tqdm.auto import tqdm

import numpy as np

import evaluate

import pandas as pd

import os

# Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Define Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", return_tensors="pt")

model = AutoModel.from_pretrained("openai-community/gpt2")

# Add Padding token if Tokenizer doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Put the dataset in a Pandas DataFrame
df = pd.read_csv('/cfs/home/u024219/Tese/CARLSy/chess_dataset.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
#df = pd.read_csv('C:\\Users\\afons\\Ambiente de Trabalho\\dataset\\chess_dataset.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
df = pd.DataFrame(df)

df['Notation:Commentary'] = df['algebraic_notation'] + ": " + df['commentary']

# Load Dataset from Pandas DataFrame
chess_dataset = Dataset.from_pandas(df)

# Separate Dataset into train, validation and test datasets
chess_dataset = chess_dataset.train_test_split(test_size= 0.1)
chess_dataset_2 = chess_dataset['test'].train_test_split(test_size=0.5)

chess_dataset = DatasetDict({
    'train': chess_dataset['train'],
    'valid': chess_dataset_2['train'],
    'test': chess_dataset_2['test']
})

# Small sample dataset if needed
#small_train_dataset = chess_dataset["train"].shuffle(seed=42).select(range(1000))
#small_eval_dataset = chess_dataset["test"].shuffle(seed=42).select(range(100))

# Tokenize the dataset
def tokenize_function(dataset):
    return tokenizer(dataset["Notation:Commentary"], truncation= True)

tokenized_dataset = chess_dataset.map(tokenize_function, batched=True)

# Remove unneeded columns 
tokenized_dataset = tokenized_dataset.remove_columns(["id","algebraic_notation", "commentary", "Notation:Commentary"])
tokenized_dataset = tokenized_dataset.with_format("torch")

print(tokenizer.convert_ids_to_tokens(tokenized_dataset["train"][0]["input_ids"]))
print(tokenized_dataset["train"][0])

# Pad the input for each batch instead of globally to save during training
data_collator = DataCollatorWithPadding(tokenizer)

train_args = Seq2SeqTrainingArguments(output_dir='./results/baidu/finetune/task22-lowdata',evaluation_strategy = 'epoch',
                                per_device_train_batch_size=2,weight_decay=0, learning_rate= 0.00005,
                                num_train_epochs=100,lr_scheduler_type='constant_with_warmup',warmup_ratio=0.1,logging_strategy='steps',
                                save_strategy='epoch',fp16_backend = 'amp',fp16 = False,gradient_accumulation_steps = 2,
                                load_best_model_at_end = True,logging_steps = 1, predict_with_generate = True)#,deepspeed='./zero2_auto_config.json', save_total_limit = 3)
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

def compute_metrics(eval_preds):
    # print(preds)
    preds, labels = eval_preds
    #print('preds:',preds[0])
    # print('len:',preds[0].shape)
    if isinstance(preds, tuple):
        preds = preds[0]
    print('preds:',preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Trainer(
    model,
    train_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
from transformers import TrainingArguments
from transformers import Seq2SeqTrainer
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

import nltk

import os

from transformers import DataCollatorForSeq2Seq

# Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Define Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

model = AutoModel.from_pretrained("google/flan-t5-small")

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
def tokenize_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = [doc for doc in examples["algebraic_notation"]]
   model_inputs = tokenizer(inputs, max_length=128, truncation=True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target=examples["commentary"], 
                      max_length=512,         
                      truncation=True)

   model_inputs["decoder_input_ids"] = labels["input_ids"]
   return model_inputs

tokenized_dataset = chess_dataset.map(tokenize_function, batched=True, remove_columns =["id","algebraic_notation", "commentary", "Notation:Commentary"])

# Remove unneeded columns 
# tokenized_dataset = tokenized_dataset.remove_columns(["id","algebraic_notation", "commentary", "Notation:Commentary"])
# tokenized_dataset = tokenized_dataset.with_format("torch")

print(tokenizer.convert_ids_to_tokens(tokenized_dataset["train"][0]["input_ids"]))
print(tokenized_dataset["train"][0])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="./results",
   evaluation_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  
   return result

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["valid"],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
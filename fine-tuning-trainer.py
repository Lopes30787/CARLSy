from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_scheduler
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

from datasets import Dataset, DatasetDict

from tqdm.auto import tqdm

import numpy as np

import evaluate

import pandas as pd

import nltk

import os

from transformers import DataCollatorForSeq2Seq

# Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]= "false"

# Define Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("/cfs/home/u024219/Tese/CARLSy/flanT5-finetuned")
tokenizer.model_max_length = 4096
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Add Padding token if Tokenizer doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Put the dataset in a Pandas DataFrame
df = pd.read_csv('/cfs/home/u024219/Tese/CARLSy/datasets/chess_dataset_extended_with_move.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
#df = pd.read_csv('C:\\Users\\afons\\Ambiente de Trabalho\\dataset\\chess_dataset.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
df = pd.DataFrame(df)
df = df.dropna()

df['Training'] ="[PGN]" + df['algebraic_notation'] + ["MOVE"] + df['move'] + "[BOARD]" + df['positions'] + "[ATTACKS] " + df['attacks'] 

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
   inputs = [doc for doc in examples["Training"]]
   model_inputs = tokenizer(inputs, max_length=4096, truncation = True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target=examples["commentary"], 
                      max_length=512,         
                      truncation=True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

tokenized_dataset = chess_dataset.map(tokenize_function, batched=True, remove_columns =["id","algebraic_notation", "commentary", "Training"])

#print(tokenizer.convert_ids_to_tokens(tokenized_dataset["train"][0]["input_ids"]))
#print(tokenized_dataset["train"][0])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 5
MAX_LENGTH = 200

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results/tokenizer-finetuned/results-extended-with-move",
    evaluation_strategy="epoch",
    learning_rate=L_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=SAVE_TOTAL_LIM,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    push_to_hub=False,
    generation_max_length=MAX_LENGTH,
    report_to="tensorboard",
    logging_dir="./tb_logs/tokenizer-extended-with-move"
)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# Rouge Metric
rouge = evaluate.load("rouge")

# Bleu Metric
bleu = evaluate.load("sacrebleu")

# Meteor Metric
meteor = evaluate.load("meteor")

def compute_meteor_rouge(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    meteor_res = meteor.compute(predictions=decoded_preds, references=decoded_labels)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  
    return meteor_res, rouge_res

def compute_bleu(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def compute_metrics(eval_preds):
    meteor, rouge = compute_meteor_rouge(eval_preds)
    bleu = compute_bleu(eval_preds)

    return {'meteor': meteor, 'rouge': rouge, 'bleu': bleu}

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

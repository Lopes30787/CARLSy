from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorWithPadding
from transformers import get_scheduler

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

#Create Dataloaders
train_dataloader = DataLoader(
    tokenized_dataset["train"], batch_size= 2, shuffle= True, collate_fn= data_collator
)

test_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size= 2, shuffle= True, collate_fn= data_collator
)

# Define optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model.to(device)
#print(f"Using device: {device}")
# Define Accelerator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader
)

# Define learning rate scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Function that takes predictions and labels and converts them to lists of strings 
def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


# Progress bar to see training progress
progress_bar = tqdm(range(num_training_steps))

metric = evaluate.load("sacrebleu")

# ======================================== #
#               Training                   #
# ======================================== #

for epoch in range(num_epochs):
    model.train()

    for batch in train_dataloader:
        #batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        # Obtain Loss
        loss = outputs[0]
    
        accelerator.backward(loss)
        #loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# ======================================== #
#              Validation                  #
# ======================================== #

    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

# ======================================== #
#                 Saving                   #
# ======================================== #

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save training arguments
# TODO: Put arguments into a variable in order to be easy to save
#torch.save(args, os.path.join(output_dir, 'training_args.bin'))
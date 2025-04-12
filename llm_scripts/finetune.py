"""
ipynb file is located at
    https://www.kaggle.com/code/realyogesh/llama3-2-1b-finetune
"""

import os
import json
import wandb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
import unsloth
from trl import SFTTrainer
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers.integrations import WandbCallback
from transformers import TrainingArguments, TextStreamer
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

with open('../dataset/eval_dset.json', 'r') as f:
    eval_dset = json.load(f)

eval_ls = []

for i in eval_dset:
  eval_ls.append([i['input'], i['reference']])
eval_df = pd.DataFrame(eval_ls, columns=['input', 'reference'])
data = eval_df

data['Context_length'] = data['input'].apply(lambda x: len(x.split(" "))) + data['reference'].apply(lambda x: len(x.split(" ")))
filtered_data = data[data['Context_length'] <= 1500]

max_seq_length = 5020
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)
print(model.print_trainable_parameters())

data_prompt = """
### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompt(examples):
    inputs       = examples["input"]
    outputs      = examples["reference"]
    texts = []
    for input_, output in zip(inputs, outputs):
        text = data_prompt.format(input_, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

training_data = Dataset.from_pandas(filtered_data.iloc[:250, :])
training_data = training_data.map(formatting_prompt, batched=True)
split_dataset = training_data.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

wandb.login(key=os.getenv("WANDB_TOKEN"))
wandb.init(project="Barz", name="llama_3.2_1b_test_finetune_2")

trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    eval_dataset=eval_dataset,
    callbacks = [WandbCallback, EarlyStoppingCallback(early_stopping_patience = 3)],
    metric_for_best_model="eval_loss",
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=50,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
        eval_strategy = IntervalStrategy.STEPS
        
    ),
)

trainer.train()
model.save_pretrained("../models/lora_model")
tokenizer.save_pretrained("../models/lora_model")
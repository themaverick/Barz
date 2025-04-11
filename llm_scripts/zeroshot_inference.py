"""
ipynb file is located at
    https://colab.research.google.com/drive/1lWBczjWs6EurAYvutDi9JYPOXpmcK4gn
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_score import score
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

with open('dataset1.json', 'r') as f:
  data_json = json.load(f)

data_ls = []

cnt = 0
for i in tqdm(data_json):
  for k in i['annotated_lyrics']:
    try:
      data_ls.append([i['song_name'], i['lyrics'], k['lyr_snip'], k['annotation']])
    except KeyError:
      continue

data_df = pd.DataFrame(data_ls, columns=['song_name', 'lyrics', 'snippet', 'annotation'])

model_id = "../models/Llama-3.2-1B-Instruct"
generator = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def create_prompt(lyrics, snippet):
    prompt = f"""
    Given the lyrics of a song and a snippet from it, break down the snippet and return its meaning.

    Lyrics:
    {lyrics}


    Snippet:
    {snippet}

    Return only the meaning, with no additional information.
    """.strip()
    return prompt

def create_dataset(df):
    rows = []
    for _, row in tqdm(data_df.iterrows()):
        rows.append(
            {
                "input": create_prompt(row.lyrics, row.snippet),
                "output": row.annotation,
            }
        )
    return rows

def batch_generator(data_rows, generator, batch_size=8, max_new_tokens=500):
    all_results = []

    for i in tqdm(range(0, len(data_rows), batch_size)):
        batch = data_rows[i:i+batch_size]

        # Prepare message format for chat models
        messages_batch = [
            [{"role": "user", "content": row["input"]}] for row in batch
        ]

        # Generate responses in batch
        outputs = generator(
            messages_batch,
            pad_token_id=generator.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )

        for row, output in zip(batch, outputs):
            all_results.append({
                "input": row["input"],
                "reference": row.get("output", ""), 
                "prediction": output[0]["generated_text"][-1]["content"]
            })

    return all_results

data_rows = create_dataset(data_df)
eval_dset = batch_generator(data_rows, generator)

data = eval_dset
references = [sample["reference"] for sample in data]
predictions = [sample["prediction"] for sample in data]

P, R, F1 = score(predictions, references, lang="en", verbose=True)

with open('../dataset/eval_dset.json', 'w') as f:
    json.dump(eval_dset, f)

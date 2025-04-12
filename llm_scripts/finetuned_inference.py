"""
ipynb file is located at
    https://www.kaggle.com/code/realyogesh/barz-finetune-inference
"""

import json
import pandas as pd
from tqdm import tqdm
import unsloth
from unsloth import FastLanguageModel

max_seq_length = 5020
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../models/lora_model",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)

data_prompt = """
### Input:
{}

### Response:
{}"""


def get_llm_output(text):
  inputs = tokenizer(
  [
      data_prompt.format(
          #instructions
          text,
          #answer
          "",
      )
  ], return_tensors = "pt").to("cuda")

  outputs = model.generate(**inputs, max_new_tokens = 5020, use_cache = True)
  answer=tokenizer.batch_decode(outputs)
  answer = answer[0].split("### Response:")[-1]
  # print("Answer of the question is:", answer)
  return answer

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


with open('../dataset/dataset1.json', 'r') as f:
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
data_rows = create_dataset(data_df)

predictions = []
true_values = []
eval_dset = []
id = 0

for i in tqdm(data_rows[:300]):
  pred = get_llm_output(i['input'])
  predictions.append(pred)
  out = i['output']
  true_values.append(out)

  eval_dset.append({
            "input": i["input"],
            "reference": out,             
            "prediction": pred.replace("<|end_of_text|>", "")
  })

with open('../dataset/eval_dset_finetuned.json', 'w') as f:
    json.dump(eval_dset, f)

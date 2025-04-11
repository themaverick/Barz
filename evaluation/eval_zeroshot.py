import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from datasets import load_dataset, Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics import SemanticSimilarity, BleuScore, RougeScore, SemanticSimilarity

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

with open('../dataset/eval_dset.json', 'r') as f:
  eval_dset = json.load(f)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

key_map = {"prediction": "response", 
        "reference": "reference"}

ragas_dset = [
    {key_map[k]: v for k, v in item.items() if k in key_map.keys()}
    for item in eval_dset
]

dataset = Dataset.from_list(ragas_dset)
eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

metric1 = FactualCorrectness(llm = evaluator_llm)
metric2 = BleuScore()
metric3 = RougeScore()
metric4 = SemanticSimilarity()

results = evaluate(eval_dataset[:300], metrics = [metric1, metric2, metric3, metric4])
results.to_pandas().to_csv(path_or_buf=  "../results/data/zeroshot.csv", index = False)
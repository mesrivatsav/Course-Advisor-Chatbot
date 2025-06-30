from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness,context_recall,context_precision,answer_relevancy
import pandas as pd
import ast

data_samples = pd.read_csv("score.csv", encoding="latin-1")

data_samples['contexts'] = data_samples['contexts'].apply(lambda x: ast.literal_eval(x)) 

print(data_samples)

dataset = Dataset.from_pandas(data_samples)

score = evaluate(dataset, metrics=[faithfulness, answer_correctness,context_recall,context_precision,answer_relevancy])
df = score.to_pandas()
df.to_csv('score_results.csv', index=False)

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this datasets
splits = {'dev': 'data/dev-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench/" + splits["dev"])

pass
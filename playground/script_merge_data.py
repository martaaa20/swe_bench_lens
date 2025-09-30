# load the swe bench verified -> and cache?
import pickle

# import pandas as pd
df = pd.read_parquet(
    "hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet"
)
df.to_pickle("./datasets/swe-bench-verified.pickle")

# load the results, merge on the issue_id -> and cache them

import pandas as pd

df = pd.read_pickle("../datasets/swe-bench-verified.pickle")

pass

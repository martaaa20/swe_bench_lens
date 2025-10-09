import codecs

from unidiff import PatchSet

from data_structures.benchmark_type_enum import BenchmarkType
from main.add_features_pipeline import AddFeaturesPipeline
from main.merge_data import BenchmarkResultsMerger

benchmark_merger = BenchmarkResultsMerger(
    BenchmarkType.VERIFIED, "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct"
)
df = benchmark_merger.get_df_with_resolved_status()
df_with_feats = AddFeaturesPipeline(input_df=df).execute()

pass
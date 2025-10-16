import codecs

from unidiff import PatchSet

from data_structures.benchmark_type_enum import BenchmarkType
from main.add_features_pipeline import AddFeaturesPipeline
from main.merge_data import BenchmarkResultsMerger

benchmark_merger = BenchmarkResultsMerger(
    BenchmarkType.VERIFIED, "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct"
)
df = benchmark_merger.get_df_with_resolved_status()
pipeline = AddFeaturesPipeline(input_df=df)
df_with_feats = pipeline.get_original_df_with_features()

df2 = pipeline.get_only_features_df()

corr_matrix = df2.corr()
print(corr_matrix)

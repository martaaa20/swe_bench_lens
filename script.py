import codecs

from unidiff import PatchSet

from data_structures.benchmark_type_enum import BenchmarkType
from main.add_features_pipeline import AddFeaturesPipeline
from main.merge_data import BenchmarkResultsMerger
from plotting import PlottingManager

agent_names = [
    "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct",
    "20250805_openhands-Qwen3-Coder-480B-A35B-Instruct",
    "20250806_SWE-Exp_DeepSeek-V3",
    "20250819_ACoder",
    "20250804_epam-ai-run-claude-4-sonnet",
]

for i, agent_name in enumerate(agent_names):
    benchmark_merger = BenchmarkResultsMerger(BenchmarkType.VERIFIED, agent_name)
    new_column_name = f"resolve_status_{agent_name}"

    if i == 0:
        result_df = benchmark_merger.get_df_with_resolved_status()
        result_df.rename(
            columns={"resolve_status": f"resolve_status_{agent_name}"}, inplace=True
        )
        result_df[f"binary_{new_column_name}"] = result_df[new_column_name].apply(
            lambda x: True if x == "resolved" else False
        )
    else:
        agent_df = benchmark_merger.get_df_with_resolved_status()
        agent_df = agent_df[["instance_id", "resolve_status"]]
        agent_df.rename(
            columns={"resolve_status": f"resolve_status_{agent_name}"}, inplace=True
        )
        agent_df[f"binary_{new_column_name}"] = agent_df[
            f"resolve_status_{agent_name}"
        ].apply(lambda x: True if x == "resolved" else False)
        result_df = result_df.join(agent_df.set_index("instance_id"), on="instance_id")


# benchmark_merger = BenchmarkResultsMerger(
#    BenchmarkType.VERIFIED, "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct"
# )
# df = benchmark_merger.get_df_with_resolved_status()
pipeline = AddFeaturesPipeline(input_df=result_df)
df_with_feats = pipeline.get_original_df_with_features()

df2 = pipeline.get_only_features_df()

corr_matrix = df2.corr()
fig = PlottingManager(df_with_feats).plot("FEAT_num_of_fail_to_pass")
fig.show()
print(corr_matrix)

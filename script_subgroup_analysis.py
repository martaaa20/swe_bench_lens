import pysubgroup as ps

from data_structures.benchmark_type_enum import BenchmarkType
from main.add_features_pipeline import AddFeaturesPipeline
from main.merge_data import BenchmarkResultsMerger

# agent_name = "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct"
agent_name = "20250805_openhands-Qwen3-Coder-480B-A35B-Instruct"
benchmark_merger = BenchmarkResultsMerger(BenchmarkType.VERIFIED, agent_name)
result_df = benchmark_merger.get_df_with_resolved_status()

result_df["binary_resolved"] = result_df["resolve_status"].apply(
    lambda x: True if x == "resolved" else False
)
result_df.drop(columns=["resolve_status"], inplace=True)

pipeline = AddFeaturesPipeline(input_df=result_df)

df = pipeline.get_df_for_subgroup_analysis()

target = ps.BinaryTarget("binary_resolved", False)
searchspace = ps.create_selectors(df, ignore=["binary_resolved"])
task = ps.SubgroupDiscoveryTask(
    df,
    target,
    searchspace,
    result_set_size=50,
    depth=5,
    qf=ps.WRAccQF(),
)

result = ps.DFS().execute(task)

for index, subgroup in enumerate(result.results):
    print(f"\n\n######## SUBGROUP {index + 1}")
    print(str(subgroup[1]))
    print("score: " + str(subgroup[0]))
    print(f"number of instances of the subgroup: {subgroup[2].size_sg}")
    print(
        f"accuracy in the SUBGROUP: {1-subgroup[2].positives_count / subgroup[2].size_sg}"
    )
    print(f"VS accuracy in the full dataset: {len(df[df["binary_resolved"]]) / 500}")
pass

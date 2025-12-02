import pysubgroup as ps

from data_structures.benchmark_type_enum import BenchmarkType
from main.add_features_pipeline import AddFeaturesPipeline
from main.merge_data import BenchmarkResultsMerger

FIND_AGENTS_STRENGTH = True


def find_subsets(
    subgroup_features,
    all_interesting_subgroups,
):
    subset_subgroups = []
    new_features = subgroup_features
    for interesting_subgroup in all_interesting_subgroups:
        old_features = interesting_subgroup[1].selectors

        if set(old_features).issubset(new_features):
            subset_subgroups.append(interesting_subgroup)

    return subset_subgroups


def is_interesting_superset(
    subgroup_features, subgroup_score, all_interesting_subgroups, baseline_accuracy
):
    results = []
    new_score = subgroup_score
    to_compare_with = find_subsets(subgroup_features, all_interesting_subgroups)
    for interesting_subgroup in to_compare_with:

        # calculate the old subgroup score
        if FIND_AGENTS_STRENGTH:
            old_score = (
                interesting_subgroup[2].positives_count
                / interesting_subgroup[2].size_sg
            )
        else:
            old_score = 1 - (
                interesting_subgroup[2].positives_count
                / interesting_subgroup[2].size_sg
            )

        # compare if the new one has a more interesting score -> keep true
        if abs(baseline_accuracy - old_score) < abs(baseline_accuracy - new_score):
            results.append(True)
        else:
            results.append(False)
    return all(results)


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

target = ps.BinaryTarget("binary_resolved", FIND_AGENTS_STRENGTH)
searchspace = ps.create_selectors(df, ignore=["binary_resolved"])
task = ps.SubgroupDiscoveryTask(
    df,
    target,
    searchspace,
    result_set_size=1000,
    depth=3,  # note: this parameter should be tested in different combinations
    qf=ps.WRAccQF(),
)

result = ps.DFS().execute(task)


# step: filter out not interesting subgroups
interesting_subgroups = []
deleted_subgroups = []
deleted_subgroups_ids = []
passed_subgroups = (
    {}
)  # dict of tuples as keys (subgroup_size, positives_in_subgroup) and redundant count

for index, subgroup in enumerate(result.results):
    size_subgroup = subgroup[2].size_sg
    target_fulfilled_count = subgroup[2].positives_count
    all_instances = subgroup[1].n_instances

    accuracy_subgroup = (
        target_fulfilled_count / size_subgroup
        if FIND_AGENTS_STRENGTH
        else 1 - (target_fulfilled_count / size_subgroup)
    )  # if target is resolved==False, the accuracy would be for False values, that's why we need to do "1-" in the beginning
    general_accuracy = len(df[df["binary_resolved"]]) / subgroup[1].n_instances

    if (
        target_fulfilled_count >= 20
        and abs(accuracy_subgroup - general_accuracy) >= 0.1
        and is_interesting_superset(
            subgroup[1].selectors,
            accuracy_subgroup,
            interesting_subgroups,
            general_accuracy,
        )
    ):
        subgroup_tuple = (size_subgroup, target_fulfilled_count)
        if subgroup_tuple in passed_subgroups:
            passed_subgroups[subgroup_tuple] += 1
        else:
            passed_subgroups[subgroup_tuple] = 1
            interesting_subgroups.append(subgroup)
    else:
        deleted_subgroups.append(subgroup)
        deleted_subgroups_ids.append(index)

# step: reranking by num of features
# (delta_accuracy, num_of_features, subgroup)
list_of_tuples_for_reranking = []
for subgroup in interesting_subgroups:
    num_of_features = len(subgroup[1].selectors)
    if FIND_AGENTS_STRENGTH:
        score = subgroup[2].positives_count / subgroup[2].size_sg
    else:
        score = 1 - (subgroup[2].positives_count / subgroup[2].size_sg)
    delta_accuracy = abs(general_accuracy - score)
    list_of_tuples_for_reranking.append((delta_accuracy, num_of_features, subgroup))

reranked_subgroups = [
    subgroup
    for _, _, subgroup in sorted(
        list_of_tuples_for_reranking, key=lambda x: (x[1], -x[0])
    )
]


# step: printing final results
for index, subgroup in enumerate(reranked_subgroups):
    print(f"\n\n######## SUBGROUP {index + 1}")
    print(str(subgroup[1]))
    print("score: " + str(subgroup[0]))
    print(f"number of instances of the subgroup: {subgroup[2].size_sg}")
    if FIND_AGENTS_STRENGTH:
        subgroup_accuracy = subgroup[2].positives_count / subgroup[2].size_sg
    else:
        subgroup_accuracy = 1 - subgroup[2].positives_count / subgroup[2].size_sg
    print(f"accuracy in the SUBGROUP: {subgroup_accuracy}")
    print(
        f"VS accuracy in the full dataset: {len(df[df["binary_resolved"]]) / subgroup[1].n_instances}"
    )
pass

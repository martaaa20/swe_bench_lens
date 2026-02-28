import math
from enum import Enum
import ast

import pysubgroup as ps
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pysubgroup import plot_distribution_numeric
from sklearn.preprocessing import MultiLabelBinarizer

from data_structures.benchmark_type_enum import BenchmarkType
from data_structures.subgroup_analysis_model import (
    SubgroupAnalysisResultModel,
    Subgroup,
    CategoryOfFeature,
)
from main.features_extraction.add_features_pipeline import AddFeaturesPipeline
from main.input_data.merge_data import BenchmarkResultsMerger


# np.seterr(divide="ignore", invalid="ignore")


class SubgroupAnalysisPipeline:
    def __init__(
        self,
        agent_name: str,
        benchmark_type: BenchmarkType,
        find_agents_strength: bool = True,
        min_subgroup_instances_ratio: float = 0.04,
    ):
        """

        :param agent_name: agent name (agent key string)
        :param benchmark_type: type of benchmark (BenchmarkType type)
        :param find_agents_strength: boolean to perform subgroup analysis based on agent strength or weaknesses
        :param min_subgroup_instances_ratio: percentage threshold (float) specifying the minimum proportion of instances required for a subgroup to be considered valid
        """
        self.agent_name = agent_name
        self.benchmark_type = benchmark_type
        self.find_agents_strength = find_agents_strength
        self.__general_accuracy = None
        self.df_with_features = None
        self.min_subgroup_instances_ratio = min_subgroup_instances_ratio

    def perform(self, depth=3, result_set_size=1000):
        # step: prepare data for the subgroup discovery
        benchmark_merger = BenchmarkResultsMerger(self.benchmark_type, self.agent_name)
        result_df = benchmark_merger.get_df_with_resolved_status()

        result_df["binary_resolved"] = result_df["resolve_status"].apply(
            lambda x: True if x == "resolved" else False
        )
        result_df.drop(columns=["resolve_status"], inplace=True)

        pipeline = AddFeaturesPipeline(input_df=result_df)
        self.df_with_features = pipeline.get_df_for_subgroup_analysis()

        # right now not using the other_programming_languages feature, commented out next liene
        # self.df_with_features = self.__get_features_df(pipeline)

        self.__general_accuracy = (
            self.df_with_features["binary_resolved"].sum()
            / self.df_with_features.shape[0]
        )

        # step: perform the subgroup discovery
        target = ps.BinaryTarget("binary_resolved", self.find_agents_strength)
        searchspace = ps.create_selectors(
            self.df_with_features,
            nbins=4,
            ignore=["binary_resolved"],
        )
        searchspace = self.__modify_searchspace(searchspace)
        task = ps.SubgroupDiscoveryTask(
            self.df_with_features,
            target,
            searchspace,
            result_set_size=result_set_size,
            depth=depth,  # note: this parameter should be tested in different combinations
            qf=ps.WRAccQF(),
        )
        result = ps.DFS().execute(task)

        # step: postprocess the subgroups
        interesting_subgroups = self.__filter_out_uninteresting_subgroups(
            result, self.df_with_features
        )

        reranked_subgroups = self.__rerank_by_num_of_features(
            interesting_subgroups, self.__general_accuracy
        )

        result_obj = self.__reformat_to_result_obj(reranked_subgroups)
        return result_obj

    def __filter_out_uninteresting_subgroups(self, result, df_with_features):
        # step: filter out not interesting subgroups
        interesting_subgroups = []
        deleted_subgroups = []
        deleted_subgroups_ids = []
        first_check_for_redundancy = (
            {}
        )  # dict of tuples as keys (subgroup_size, positives_in_subgroup) and tuple of (all_instances, positive_instances) as values

        for index, subgroup in enumerate(result.results):
            size_subgroup = subgroup[2].size_sg
            target_fulfilled_count = subgroup[2].positives_count

            accuracy_subgroup = (
                target_fulfilled_count / size_subgroup
                if self.find_agents_strength
                else 1 - (target_fulfilled_count / size_subgroup)
            )  # if target is resolved==False, the accuracy would be for False values, that's why we need to do "1-" in the beginning
            general_accuracy = (
                len(df_with_features[df_with_features["binary_resolved"]])
                / subgroup[1].n_instances
            )

            min_required_count = math.floor(
                len(self.df_with_features) * self.min_subgroup_instances_ratio
            )  # min num_instances required is min_subgroup_instances_ratio (default=4%) of the dataset

            # step: filter out the subgroups if one of the following applies: 1) too little instances in the subgroup;
            #  2) delta of accuracy between the general and the subgroup one is too small; 3) is a less interesting subset
            #  (described detailed in my paper in: Methodology -> Heuristic elimination -> Sttep 4)
            if (
                target_fulfilled_count >= min_required_count
                and abs(accuracy_subgroup - general_accuracy) >= 0.1
                and self.__is_interesting_superset(
                    subgroup[1].selectors,
                    accuracy_subgroup,
                    interesting_subgroups,
                    general_accuracy,
                )
            ):

                all_instance_ids, positive_instance_ids = (
                    self.__helper_get_instance_ids_of_subgroup(subgroup)
                )

                instances_tuple_for_redundancy = (
                    all_instance_ids,
                    positive_instance_ids,
                )

                # step: Eliminate duplicate subgroups with identical coverage and identical positive instance sets.
                #   Keep only the first occurrence.

                subgroup_tuple = (size_subgroup, target_fulfilled_count)
                redundant = False
                if (
                    subgroup_tuple in first_check_for_redundancy
                ):  # removing the subgroups that have the same exact instances in the subgroup (aka removing duplicates)
                    for similar_subgroup in first_check_for_redundancy[subgroup_tuple]:
                        if set(similar_subgroup[0]) == set(
                            instances_tuple_for_redundancy[0]
                        ):
                            if set(similar_subgroup[1]) == set(
                                instances_tuple_for_redundancy[1]
                            ):
                                redundant = True

                if not redundant:
                    interesting_subgroups.append(subgroup)

                    # step: adding the subgroup as a new subgroup for next groups' redundancy checks
                    if subgroup_tuple not in first_check_for_redundancy:
                        first_check_for_redundancy[subgroup_tuple] = []

                    first_check_for_redundancy[subgroup_tuple].append(
                        instances_tuple_for_redundancy
                    )

                else:
                    deleted_subgroups.append(subgroup)
                    deleted_subgroups_ids.append(index)

            else:
                deleted_subgroups.append(subgroup)
                deleted_subgroups_ids.append(index)

        return interesting_subgroups

    def __rerank_by_num_of_features(self, interesting_subgroups, general_accuracy):
        # (delta_accuracy, num_of_features, subgroup)
        list_of_tuples_for_reranking = []
        for subgroup in interesting_subgroups:
            num_of_features = len(subgroup[1].selectors)
            if self.find_agents_strength:
                score = subgroup[2].positives_count / subgroup[2].size_sg
            else:
                score = 1 - (subgroup[2].positives_count / subgroup[2].size_sg)
            delta_accuracy = abs(general_accuracy - score)
            list_of_tuples_for_reranking.append(
                (delta_accuracy, num_of_features, subgroup)
            )

        reranked_subgroups = [
            subgroup
            for _, _, subgroup in sorted(
                list_of_tuples_for_reranking, key=lambda x: (x[1], -x[0])
            )
        ]
        return reranked_subgroups

    def __reformat_to_result_obj(self, subgroups_calculated):
        sg_discovery_result = SubgroupAnalysisResultModel(
            subgroups=[],
            overall_accuracy=self.__general_accuracy,
            overall_instances_count=subgroups_calculated[0][1].n_instances,
        )

        subgroups_list = []
        for subgroup in subgroups_calculated:

            if self.find_agents_strength:
                sg_accuracy = subgroup[2].positives_count / subgroup[2].size_sg
            else:
                sg_accuracy = 1 - subgroup[2].positives_count / subgroup[2].size_sg

            sg_elem = Subgroup(
                selector_str=str(subgroup[1]),
                selectors=subgroup[1].selectors,
                num_instances=subgroup[2].size_sg,
                subgroup_accuracy=sg_accuracy,
                subgroup_interestingness=subgroup[0],
            )
            subgroups_list.append(sg_elem)

        sg_discovery_result.subgroups = subgroups_list

        return sg_discovery_result

    def print_results(self, subgroup_discovery_result):
        subgroup_discovery_result.pretty_print()

    def get_data(self, subgroup_discovery_result):
        return subgroup_discovery_result.get_data()

    def get_list_of_features(self):
        if self.df_with_features is None:
            self.perform()
        all_cols = list(self.df_with_features.columns)
        feature_cols = all_cols.copy()
        feature_cols.remove("binary_resolved")
        return feature_cols

    def get_dict_of_features_and_categories(self):
        feature_to_category = {
            "FEAT_created_n_months_ago": CategoryOfFeature.REPOSITORY,
            "FEAT_files_hierarchy_delta": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_latest_commit_in_repo_n_days_ago": CategoryOfFeature.REPOSITORY,
            "FEAT_length_of_description": CategoryOfFeature.ISSUE_DESCRIPTION,
            "FEAT_num_of_fail_to_pass": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_num_of_pass_to_pass": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_num_of_hunks": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_num_of_files_changed": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_patch_spread": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_num_of_modified_lines": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_num_of_deletions": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_num_of_additions": CategoryOfFeature.GROUND_TRUTH,
            "FEAT_num_of_code_mentions": CategoryOfFeature.ISSUE_DESCRIPTION,
            "FEAT_num_of_contributors_repo": CategoryOfFeature.REPOSITORY,
            "FEAT_primary_language_repo": CategoryOfFeature.REPOSITORY,
            "FEAT_number_of_files_in_repo": CategoryOfFeature.REPOSITORY,
            "FEAT_repo_size_in_kb": CategoryOfFeature.REPOSITORY,
            "FEAT_num_of_stars_repo": CategoryOfFeature.REPOSITORY,
            "FEAT_repository_name": CategoryOfFeature.REPOSITORY,
        }
        # note: the other_programming_languages feature is one-hot encoded -> could potentially many columns,
        #  we cannot manually add here right now -> not in this dict

        result_feature_to_category = {}
        for feat_column in self.get_list_of_features():
            # check if the column has an assigned category
            assert feat_column in feature_to_category.keys() or feat_column.startswith(
                "FEAT_other_languages_repo_"
            ), (
                "The feature is unknown, please assign the category in this function",
                feat_column,
            )

            if feat_column.startswith("FEAT_other_languages_repo_"):
                result_feature_to_category[feat_column] = CategoryOfFeature.REPOSITORY
            else:
                result_feature_to_category[feat_column] = feature_to_category[
                    feat_column
                ]

        return result_feature_to_category

    def __modify_searchspace(self, searchspace):
        """
        deletes some unnecessary selectors
        """
        # step: delete the selectors that cover 95% of the instances
        filtered_searchspace = []
        for sel in searchspace:
            mask = sel.covers(self.df_with_features)
            coverage = mask.mean()
            if coverage <= 0.95:
                filtered_searchspace.append(sel)

        filtered_searchspace = [
            s
            for s in filtered_searchspace
            if not (
                hasattr(s, "attribute_name")
                and s.attribute_name.startswith("FEAT_other_languages_repo")
                and str(s).endswith("==0")
            )
        ]

        return filtered_searchspace

    @staticmethod
    def __get_features_df(pipeline):
        df = pipeline.get_df_for_subgroup_analysis()

        # prepare the df for subgroup analysis, convert secondary_languages list to hot-one encodings ------------------
        df["FEAT_other_languages_repo"] = df["FEAT_other_languages_repo"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Create one-hot encoding for the list column
        mlb = MultiLabelBinarizer()
        language_encoded = mlb.fit_transform(df["FEAT_other_languages_repo"])
        language_df = pd.DataFrame(
            language_encoded,
            columns=[f"FEAT_other_languages_repo_{lang}" for lang in mlb.classes_],
            index=df.index,
        )

        # Concatenate with original df
        df = pd.concat([df, language_df], axis=1)

        # Optionally drop the original column
        df = df.drop("FEAT_other_languages_repo", axis=1)
        return df

    @staticmethod
    def __find_subsets(
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

    def __is_interesting_superset(
        self,
        subgroup_features,
        subgroup_score,
        all_interesting_subgroups,
        baseline_accuracy,
    ):
        results = []
        new_score = subgroup_score
        to_compare_with = SubgroupAnalysisPipeline.__find_subsets(
            subgroup_features, all_interesting_subgroups
        )
        for interesting_subgroup in to_compare_with:

            # calculate the old subgroup score
            if self.find_agents_strength:
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

    def __helper_get_instance_ids_of_subgroup(self, subgroup):
        mask = subgroup[1].covers(self.df_with_features)
        all_instance_ids = self.df_with_features.index[mask].to_list()

        positive_mask = mask & (self.df_with_features["binary_resolved"] == True)
        positive_instance_ids = self.df_with_features.index[positive_mask].to_list()
        return all_instance_ids, positive_instance_ids


# agent_name = "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct"
# agent_name = "20250805_openhands-Qwen3-Coder-480B-A35B-Instruct"
agent_name1 = "20251120_livesweagent_gemini-3-pro-preview"
agent_name2 = "20251103_sonar-foundation-agent_claude-sonnet-4-5"
benchmark_merger = BenchmarkResultsMerger(BenchmarkType.VERIFIED, agent_name2)

if __name__ == "__main__":
    sg_disc1 = SubgroupAnalysisPipeline(agent_name1, BenchmarkType.VERIFIED, True)
    subgroups1 = sg_disc1.perform()
    sg_disc1.print_results(subgroups1)

    sg_disc2 = SubgroupAnalysisPipeline(agent_name1, BenchmarkType.VERIFIED, False)
    subgroups2 = sg_disc2.perform()
    sg_disc2.print_results(subgroups2)
    pass

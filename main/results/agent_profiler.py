from collections import Counter, defaultdict
from typing import List, Dict

import pandas as pd
from pydantic import BaseModel

from data_structures.benchmark_type_enum import BenchmarkType
from main.subgroup_analysis.subgroup_analysis_pipeline import (
    Subgroup,
    SubgroupAnalysisResultModel,
    SubgroupAnalysisPipeline,
    CategoryOfFeature,
)


class CategoryAnalysis(BaseModel):
    category_name: CategoryOfFeature
    total_occurrences: int
    normalized_score: float


class AgentProfiler:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def create_profile(
        self,
        positive_result: SubgroupAnalysisResultModel,
        negative_result: SubgroupAnalysisResultModel,
        feature_categories: Dict[str, str],
        printable=False,
    ):

        print(f"---------- Agent: {self.agent_name}----------\n")

        self.analyze_top_performance_changes(
            positive_result, negative_result, printable=printable
        )
        self.analyze_largest_subgroups(
            positive_result, negative_result, printable=printable
        )
        self.analyze_highest_interestingness(
            positive_result, negative_result, printable=printable
        )

        all_subgroups = positive_result.subgroups + negative_result.subgroups
        self.profile_feature_ranges(all_subgroups, printable=printable)

        self.rank_feature_categories(
            all_subgroups, feature_categories, printable=printable
        )

        self.rank_features_in_each_category(
            all_subgroups, feature_categories, printable=printable
        )

        self.rank_overall_features(all_subgroups, printable=printable)

    def analyze_top_performance_changes(
        self,
        performance_increased_subgroups: SubgroupAnalysisResultModel,
        performance_drop_subgroups: SubgroupAnalysisResultModel,
        printable: bool = False,
    ):
        """
        1) Biggest performance drop/increase (TOP 3)
        """
        increased_deltas = [
            (
                sg,
                sg.subgroup_accuracy - performance_increased_subgroups.overall_accuracy,
            )
            for sg in performance_increased_subgroups.subgroups
        ]
        dropped_deltas = [
            (sg, sg.subgroup_accuracy - performance_drop_subgroups.overall_accuracy)
            for sg in performance_drop_subgroups.subgroups
        ]

        top_increases = sorted(increased_deltas, key=lambda x: x[1], reverse=True)[:20]
        top_drops = sorted(dropped_deltas, key=lambda x: x[1])[:20]

        rows = []

        for sg, delta in top_increases:
            rows.append(
                {
                    "Type": "Increase",
                    "Delta": float(delta),
                    "Acc": sg.subgroup_accuracy,
                    "Num of instances": sg.num_instances,
                    "Selector": sg.selector_str,
                }
            )

        for sg, delta in top_drops:
            rows.append(
                {
                    "Type": "Drop",
                    "Delta": float(delta),
                    "Acc": sg.subgroup_accuracy,
                    "Num of instances": sg.num_instances,
                    "Selector": sg.selector_str,
                }
            )

        df = pd.DataFrame(rows)
        if printable:
            print("--- 1. Top Performance Changes ---")
            print(df.to_string(index=False))
            print("\n")
        return df

    def analyze_largest_subgroups(
        self,
        performance_increased_subgroups: SubgroupAnalysisResultModel,
        performance_drop_subgroups: SubgroupAnalysisResultModel,
        printable: bool = False,
    ):
        """
        2) Biggest num of instances in the subgroups
        """
        all_sg = (
            performance_increased_subgroups.subgroups
            + performance_drop_subgroups.subgroups
        )
        sorted_sg = sorted(all_sg, key=lambda x: x.num_instances, reverse=True)[:20]

        rows = []
        for sg in sorted_sg:
            rows.append(
                {
                    "Instances": sg.num_instances,
                    "Acc": sg.subgroup_accuracy,
                    "Selector": sg.selector_str,
                }
            )

        df = pd.DataFrame(rows)

        if printable:
            print(
                "--- 2. LARGEST SUBGROUPS (NUM OF INSTANCES) -------------------------------------------------------"
            )
            print(df.to_string(index=False))
            print("\n")
        return df

    def analyze_highest_interestingness(
        self,
        performance_increased_subgroups: SubgroupAnalysisResultModel,
        performance_drop_subgroups: SubgroupAnalysisResultModel,
        printable: bool = False,
    ):
        """
        3) Highest interestingness.
        """
        scored_sgs = []

        for sg in performance_increased_subgroups.subgroups:
            delta = (
                sg.subgroup_accuracy - performance_increased_subgroups.overall_accuracy
            )
            score = sg.num_instances * abs(delta)
            scored_sgs.append((sg, delta, score))

        for sg in performance_drop_subgroups.subgroups:
            delta = sg.subgroup_accuracy - performance_drop_subgroups.overall_accuracy
            score = sg.num_instances * abs(delta)
            scored_sgs.append((sg, delta, score))

        top_interestingness = sorted(scored_sgs, key=lambda x: x[2], reverse=True)[:20]

        rows = []
        for sg, delta, score in top_interestingness:
            rows.append(
                {
                    "Score": float(score),
                    "Instances": sg.num_instances,
                    "Delta": float(delta),
                    "Selector": sg.selector_str,
                }
            )

        df = pd.DataFrame(rows)

        if printable:
            print(
                "----- 3: HIGHEST INTERESTINGNESS ------------------------------------------------------------------"
            )
            print(df.to_string(index=False))
            print("\n")
        return df

    def profile_feature_ranges(
        self, subgroups: List[Subgroup], printable: bool = False
    ):
        """
        4) Get feature ranges
        """
        feature_ranges = defaultdict(set)

        for sg in subgroups:
            for selector in sg.selectors:
                attr = getattr(selector, "attribute_name", "Unknown")

                selector_str = str(selector)
                feature_ranges[attr].add(selector_str)

        if printable:
            print(
                "----- 4: SELECTORS RANGES -------------------------------------------------------------------------"
            )
            for attr, ranges in feature_ranges.items():
                range_list = ", ".join(sorted(list(ranges)))
                print(f"Feature: {attr}")
                print(f"  Active Ranges: {range_list}\n")
            print("\n")
        return feature_ranges

    def rank_feature_categories(
        self,
        subgroups: List[Subgroup],
        feature_categories: Dict[str, str],
        printable: bool = False,
    ):
        """
        5) Which feature category occurs the most often.
        """
        cat_occurrences = Counter()

        category_counts = Counter(feature_categories.values())

        for sg in subgroups:
            for selector in sg.selectors:
                attr = getattr(selector, "attribute_name", "Unknown")
                cat = feature_categories.get(attr, "Uncategorized")
                cat_occurrences[cat] += 1

        results = []
        for cat, count in cat_occurrences.items():
            total_feats = category_counts.get(cat, 1)
            if total_feats == 0:
                total_feats = 1

            # normalization: (times appeared) / (number of features in that category)
            norm_score = count / total_feats
            results.append(
                CategoryAnalysis(
                    category_name=cat,
                    total_occurrences=count,
                    normalized_score=norm_score,
                )
            )

        results.sort(key=lambda x: x.normalized_score, reverse=True)

        rows = []
        for r in results:
            total = category_counts.get(r.category_name, 1)
            rows.append(
                {
                    "Category": r.category_name.value,
                    "Occurrences": r.total_occurrences,
                    "Total number of features": total,
                    "Score (avg. occurrences per feature in category)": round(
                        r.normalized_score, 2
                    ),
                }
            )

        df = pd.DataFrame(rows)
        if printable:
            print("--- 5. Feature Category Analysis (Normalized) ---")
            print(df.to_string(index=False))
            print("\n")

        return df

    def rank_features_in_each_category(
        self,
        subgroups: List[Subgroup],
        feature_categories: Dict[str, str],
        printable: bool = False,
    ):
        """
        6) Ranking of the features based on their category.
        """

        feature_counts = Counter()
        for sg in subgroups:
            for selector in sg.selectors:
                attr = getattr(selector, "attribute_name", "Unknown")
                feature_counts[attr] += 1

        # group by category
        by_category = defaultdict(list)
        for attr, count in feature_counts.items():
            cat = feature_categories.get(attr, "Uncategorized")
            by_category[cat].append((attr, count))

        # save the results to a dataframe
        results = {}

        for cat, feats in by_category.items():
            rows = []
            sorted_feats = sorted(feats, key=lambda x: x[1], reverse=True)

            for attr, count in sorted_feats:
                rows.append(
                    {
                        "Category": cat,
                        "Feature": attr,
                        "Occurrences": count,
                    }
                )

            df = pd.DataFrame(rows)
            results[cat.value] = df

        if printable:
            print("--- 6. Feature Ranking by Category ---")

            for category_name, result_df in results.items():
                print(f"--------- {category_name} --------- ")
                print(result_df.to_string(index=False))
                print("\n")
            print("\n")
        return results

    def rank_overall_features(self, subgroups: List[Subgroup], printable: bool = False):
        """
        7) Overall ranking of the features.
        """
        feature_counts = Counter()
        for sg in subgroups:
            for selector in sg.selectors:
                attr = getattr(selector, "attribute_name", "Unknown")
                feature_counts[attr] += 1

        most_common = feature_counts.most_common()

        df = pd.DataFrame(most_common, columns=["Feature", "Occurrences"])
        df.insert(0, "Rank", range(1, len(df) + 1))

        if printable:
            print(
                "----- 7: OVERALL FEATURE RANKING ------------------------------------------------------------------"
            )
            pd.set_option("display.colheader_justify", "left")
            print(df.to_string(index=False))
        return df


if __name__ == "__main__":
    agent_name = "20250805_openhands-Qwen3-Coder-480B-A35B-Instruct"
    subgroup_analysis_pipeline_strengths = SubgroupAnalysisPipeline(
        agent_name, BenchmarkType.VERIFIED, True
    )

    subgroup_analysis_pipeline_weaknesses = SubgroupAnalysisPipeline(
        agent_name, BenchmarkType.VERIFIED, False
    )

    subgroups_for_performance_increase = subgroup_analysis_pipeline_strengths.perform()
    subgroups_for_performance_drop = subgroup_analysis_pipeline_weaknesses.perform()

    features_to_categories = (
        subgroup_analysis_pipeline_strengths.get_dict_of_features_and_categories()
    )

    agent_profiler = AgentProfiler(agent_name)
    agent_profiler.create_profile(
        subgroups_for_performance_increase,
        subgroups_for_performance_drop,
        features_to_categories,
        True,
    )

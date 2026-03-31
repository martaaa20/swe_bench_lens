import sys

import numpy as np
import pandas as pd
from numpy import ndarray

from data_structures.benchmark_type_enum import BenchmarkType
from main.features_extraction.add_features_pipeline import AddFeaturesPipeline
from main.input_data.benchmark_downloder import BenchmarkDownloader
from main.input_data.merge_data import BenchmarkResultsMerger
from main.subgroup_analysis.subgroup_analysis_pipeline import SubgroupAnalysisPipeline


class ExplorataryAgentSubgroupComparator:

    def __init__(
        self,
        main_agent: str,
        other_agents: list[str],
        find_agents_strengths: bool,
        benchmark_type: BenchmarkType = BenchmarkType.VERIFIED,
    ):
        """

        :param main_agent: The name identifier of the main agent to perform subgroup analysis on
        :param other_agents: Other agents' name identifiers that the main agents should be compared to
        benchmark_type: The type of the benchmark to evaluate the agent on
        """
        self.main_agent = main_agent
        self.other_agents = other_agents
        self.find_agents_strengths = find_agents_strengths
        self.benchmark_type = benchmark_type

    @staticmethod
    def prettify_accuracy(float_number, delta_format=False):

        result = ""
        if delta_format:
            result += "+" if not str(float_number).startswith("-") else ""

        result += str(round(float_number * 100, 1))
        result += "%"

        return result

    @staticmethod
    def get_agent_str(agent_number: int, accuracy: float) -> str:
        accuracy_prettified = ExplorataryAgentSubgroupComparator.prettify_accuracy(
            accuracy
        )
        return f"Agent{agent_number} (acc. = {accuracy_prettified})"

    def run(self):
        subgroup_discovery_pipeline = SubgroupAnalysisPipeline(
            self.main_agent, self.benchmark_type, self.find_agents_strengths
        )
        results_main_agent = subgroup_discovery_pipeline.perform()

        overall_acc = results_main_agent.overall_accuracy

        other_agents_dfs = self.__get_dfs_other_agents()

        for subgroup_idx, subgroup in enumerate(results_main_agent.subgroups):
            subgroup_acc = subgroup.subgroup_accuracy
            delta_acc = subgroup_acc - overall_acc
            delta_acc_str = (
                "+"
                if not str(delta_acc).startswith("-")
                else "" + str(round(delta_acc * 100, 1)) + "%"
            )

            # calculate mask for what subgroup covers
            num_benchmark_rows = len(
                BenchmarkDownloader().get_dataset_as_df(BenchmarkType.VERIFIED)
            )
            mask = np.ones(num_benchmark_rows, dtype=bool)
            for selector in subgroup.selectors:
                mask = mask & selector.representation

            dict_for_df = {
                "Agent": [self.get_agent_str(1, overall_acc)],
                "Delta": [self.prettify_accuracy(delta_acc, delta_format=True)],
                "Accuracy": [self.prettify_accuracy(subgroup_acc)],
            }

            for i, other_agent_name in enumerate(self.other_agents):

                # prepare the dataframe for the other agent
                df_with_features = other_agents_dfs[other_agent_name]

                filtered_df = df_with_features[mask]
                other_agent_accuracy = df_with_features["binary_resolved"].mean()
                other_agent_subgroup_accuracy = filtered_df["binary_resolved"].mean()

                other_agent_delta_accuracy = (
                    other_agent_subgroup_accuracy - other_agent_accuracy
                )

                dict_for_df["Agent"].append(
                    self.get_agent_str(i + 2, other_agent_accuracy)
                )
                dict_for_df["Delta"].append(
                    self.prettify_accuracy(
                        other_agent_delta_accuracy, delta_format=True
                    )
                )
                dict_for_df["Accuracy"].append(
                    self.prettify_accuracy(other_agent_subgroup_accuracy)
                )

            self.__print_results_for_subgroup(subgroup, subgroup_idx, dict_for_df)

    def __print_results_for_subgroup(
        self, subgroup_instance, subgroup_idx, dict_comparison_with_other_agents
    ):

        df = pd.DataFrame(dict_comparison_with_other_agents)
        markdown_table = df.to_markdown(index=False)

        print(f"\n\n######## SUBGROUP {subgroup_idx + 1}")
        print(subgroup_instance.selector_str)
        print("score: " + str(subgroup_instance.subgroup_interestingness))
        print(f"number of instances of the subgroup: {subgroup_instance.num_instances}")
        print(f"accuracy in the SUBGROUP: {subgroup_instance.subgroup_accuracy}")

        print(f"comparison between agents:\n")
        print(markdown_table)

    def __get_dfs_other_agents(self):
        result = {}
        for other_agent_name in self.other_agents:
            benchmark_merger = BenchmarkResultsMerger(
                self.benchmark_type, other_agent_name
            )
            result_df = benchmark_merger.get_df_with_resolved_status()
            result_df["binary_resolved"] = result_df["resolve_status"].apply(
                lambda x: True if x == "resolved" else False
            )
            result_df.drop(columns=["resolve_status"], inplace=True)
            pipeline = AddFeaturesPipeline(input_df=result_df)
            df_with_features = pipeline.get_df_for_subgroup_analysis()

            result[other_agent_name] = df_with_features

        return result


if __name__ == "__main__":
    agent_name1 = "20251120_livesweagent_gemini-3-pro-preview"
    agent_name2 = "20251103_sonar-foundation-agent_claude-sonnet-4-5"
    agent_name3 = "20250819_ACoder"
    ExplorataryAgentSubgroupComparator(
        main_agent=agent_name1,
        other_agents=[agent_name2, agent_name3],
        find_agents_strengths=False,
    ).run()

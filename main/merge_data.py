import pandas as pd

from data_structures.benchmark_type_enum import BenchmarkType
from data_structures.resolve_status_enum import ResolveStatusEnum
from main.benchmark_downloder import BenchmarkDownloader
from main.result_data_processor import ResultDataProcessor


class BenchmarkResultsMerger:
    def __init__(self, benchmark_type, agent_eval_name):
        self.benchmark_type = benchmark_type
        self.agent_eval_name = agent_eval_name
        self.df_with_resolved_status = self.__get_df_with_resolved_status()

    def get_df_with_resolved_status(self):
        return self.df_with_resolved_status

    def __get_df_with_resolved_status(self):
        dataset_downloader = BenchmarkDownloader()
        result_data_processor = ResultDataProcessor(
            self.benchmark_type, self.agent_eval_name
        )
        agent_results_json = result_data_processor.get_results_json()

        # check if new keys in the results.json
        if not (
            missing_keys := (
                set(agent_results_json.keys())
                - {elem.value for elem in ResolveStatusEnum}
            )
        ):
            raise ValueError(
                f"New ResolveStatusEnum value(s) needed: {list(missing_keys)}"
            )
        # step: add the resole_status column with the right results
        resolved_instance_ids = agent_results_json["resolved"]
        no_logs_instance_ids = agent_results_json["no_logs"]
        no_generated_instance_ids = agent_results_json["no_generation"]

        benchmark_df = dataset_downloader.get_dataset_as_df(self.benchmark_type)
        benchmark_df["resolve_status"] = pd.NA
        benchmark_df.loc[
            benchmark_df["instance_id"].isin(resolved_instance_ids), "resolve_status"
        ] = ResolveStatusEnum.RESOLVED

        benchmark_df.loc[
            benchmark_df["instance_id"].isin(no_logs_instance_ids), "resolve_status"
        ] = ResolveStatusEnum.ERR_NO_LOGS

        benchmark_df.loc[
            benchmark_df["instance_id"].isin(no_generated_instance_ids),
            "resolve_status",
        ] = ResolveStatusEnum.ERR_NO_GENERATION

        benchmark_df.fillna(
            {"resolve_status": ResolveStatusEnum.NOT_RESOLVED}, inplace=True
        )

        return benchmark_df

    @staticmethod
    def add_features(input_df):
        input_df = input_df.copy()
        input_df = BenchmarkResultsMerger.__add_num_of_fail_to_pass(input_df)
        input_df = BenchmarkResultsMerger.__add_num_of_hunks(input_df)

        return input_df

    def get_full_stats(self) -> dict:
        """
        :return: a dictionary showing all the categories percentages (values from0 until 1), e.g. {"NOT_RESOLVED": 0.4, "RESOLVED": 0.6}
        """
        df = self.df_with_resolved_status
        full_stats_dict = {}
        for resolve_status in ResolveStatusEnum:
            resolve_status_str = resolve_status.value
            sub_df = df.loc[df["resolve_status"] == resolve_status_str]
            percentage = len(sub_df) / len(df) * 100
            full_stats_dict[resolve_status_str] = percentage
        return full_stats_dict

    def get_resolved_stats(self) -> float:
        """
        :return: a float between 0 and 100 % showing how many instances have been resolved by the agent
        """
        df = self.df_with_resolved_status
        resolved_sub_df = df.loc[df["resolve_status"] == "resolved"]
        resolved_percentage = len(resolved_sub_df) / len(df) * 100
        return resolved_percentage


if __name__ == "__main__":
    benchmark_merger = BenchmarkResultsMerger(
        BenchmarkType.VERIFIED, "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct"
    )
    result = benchmark_merger.get_resolved_stats()
    result2 = benchmark_merger.get_full_stats()
    print(result)
    print(result2)

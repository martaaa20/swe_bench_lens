import json
from pathlib import Path

from data_structures.benchmark_type_enum import BenchmarkType


class ResultDataProcessor:
    path_to_data = Path("C:\code\swe-bench\data")

    def __init__(self, benchmark_type: BenchmarkType, agent_eval_name: str):
        self.benchmark_type = benchmark_type
        self.agent_eval_name = agent_eval_name
        self.path_to_agents_dir = (
            ResultDataProcessor.path_to_data
            / self.benchmark_type.value
            / self.agent_eval_name
        )

    def __get_results_file(self, filename: str):
        with open(self.path_to_agents_dir / "results" / f"{filename}.json", "r") as f:
            results_json = json.load(f)
        return results_json

    def get_results_json(
        self,
    ):
        return self.__get_results_file("results")

    def get_results_by_time_json(self):
        return self.__get_results_file("results_by_time")

    def get_results_by_repo_json(self):
        return self.__get_results_file("results_by_repo")

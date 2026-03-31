from pathlib import Path
from os import listdir
from os.path import isfile, join

import pandas as pd

from data_structures.benchmark_type_enum import BenchmarkType


class BenchmarkDownloader:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    datasets_path = PROJECT_ROOT / "datasets"
    if not datasets_path.exists():
        datasets_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def download_all_datasets() -> None:
        """Downloads all the datasets, should be executed when a new dataset is needed."""
        path = BenchmarkDownloader.datasets_path
        downloaded_datasets = [
            f.split(".pickle")[0] for f in listdir(path) if isfile(join(path, f))
        ]

        for bench_type in BenchmarkType:
            benchmark_filename = bench_type.value
            if benchmark_filename not in downloaded_datasets:
                BenchmarkDownloader.__download_dataset(bench_type)

    @staticmethod
    def __download_dataset(benchmark: BenchmarkType):
        if benchmark == BenchmarkType.VERIFIED:
            df = pd.read_parquet(
                "hf://datasets/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet"
            )
        elif benchmark == BenchmarkType.LITE:
            splits = {
                "dev": "data/dev-00000-of-00001.parquet",
                "test": "data/test-00000-of-00001.parquet",
            }
            df = pd.read_parquet(
                "hf://datasets/princeton-nlp/SWE-bench_Lite/" + splits["dev"]
            )
        else:
            raise ValueError(
                f"Invalid benchmark type: {benchmark}; no implementation for downloading this dataset. Add code to download "
                f"in the benchmark_downloader.py file."
            )
        df.to_pickle(BenchmarkDownloader.datasets_path / f"{benchmark.value}.pickle")

    def get_dataset_as_df(self, benchmark: BenchmarkType) -> pd.DataFrame:
        downloaded_datasets = [
            f.split(".pickle")[0]
            for f in listdir(self.datasets_path)
            if isfile(join(self.datasets_path, f))
        ]

        if benchmark.value not in downloaded_datasets:
            self.__download_dataset(benchmark)

        return pd.read_pickle(self.datasets_path / f"{benchmark.value}.pickle")


if __name__ == "__main__":
    BenchmarkDownloader.download_all_datasets()

import pandas as pd
import pysubgroup as ps

from data_structures.benchmark_type_enum import BenchmarkType
from main.features_extraction.add_features_pipeline import AddFeaturesPipeline
from main.input_data.merge_data import BenchmarkResultsMerger

import plotly.express as px


class HistogramsForQuantiles:
    def __init__(self, benchmark_type, agent):
        self.benchmark_type = benchmark_type
        self.agent_name = agent
        benchmark_merger = BenchmarkResultsMerger(self.benchmark_type, self.agent_name)
        df = benchmark_merger.get_df_with_resolved_status()

        df["binary_resolved"] = df["resolve_status"].apply(
            lambda x: True if x == "resolved" else False
        )
        df.drop(columns=["resolve_status"], inplace=True)

        pipeline = AddFeaturesPipeline(input_df=df)
        self.df_with_features = pipeline.get_df_for_subgroup_analysis()
        self.features_ls = list(self.df_with_features.columns)
        self.features_ls.remove("binary_resolved")

    def cli(self, quantiles=None):
        """

        :param quantiles: if you want to choose the quantiles in the CLI, leave it as None. Otherwise, put the number of quantiles for the histogram.

        """
        while True:
            print("which feature do you want to use?")

            features_str = "\n".join(
                [f"[{i}] {x}" for i, x in enumerate(self.features_ls)]
            )
            print(features_str)
            index_of_feature_chosen = int(input("Select the number: "))
            feature_chosen = self.features_ls[index_of_feature_chosen]

            # find out is it nominal or numerical
            test_selectors_chosen_feature = ps.create_selectors(
                self.df_with_features[[feature_chosen]]
            )
            feature_type = self.get_nominal_or_numerical(test_selectors_chosen_feature)

            # if it's nominal only create the selectors
            selectors = None
            if feature_type == "nominal":
                selectors = test_selectors_chosen_feature

            elif feature_type == "numerical":
                if quantiles is None:
                    quantiles_input = int(input("Choose the number of quantiles: "))
                selectors = ps.create_selectors(
                    self.df_with_features[[feature_chosen]],
                    nbins=quantiles or quantiles_input,
                )
            else:
                raise ValueError(f"Invalid feature type: {feature_type}")

            data_for_histogram = []
            target = ps.BinaryTarget("binary_resolved", True)
            for i, selector in enumerate(selectors):
                data = {}

                df = self.df_with_features.copy()

                description = ps.subgroup_description.Conjunction([selector])
                cover = description.covers(df)

                subgroup = df.loc[cover, "binary_resolved"]

                # Compute accuracy: proportion of positives
                accuracy = subgroup.mean()
                print(accuracy)
                print(sum(cover))

                data.update(accuracy=accuracy)
                data.update(num_instances=sum(cover))
                data.update(
                    description=str(description)
                    .strip()
                    .strip(feature_chosen)
                    .strip(":")
                    .strip()
                )
                data_for_histogram.append(data)

            df = pd.DataFrame.from_records(data_for_histogram)
            print(df)
            self.create_plot(df, feature_chosen)

    def get_nominal_or_numerical(self, test_selectors_for_feature) -> str:
        if isinstance(test_selectors_for_feature[0], ps.EqualitySelector):
            return "nominal"
        elif isinstance(test_selectors_for_feature[0], ps.IntervalSelector):
            return "numerical"
        else:
            raise ValueError(
                "The test selectors should be of type ps.EqualitySelector or ps.IntervalSelector"
            )

    def create_plot(self, df, feature_name):
        df["accuracy_rounded"] = df["accuracy"].map(lambda x: f"{x * 100:.1f}%")

        fig = px.bar(
            df,
            x="description",
            y="accuracy",
            text="accuracy_rounded",
            color_discrete_sequence=["steelblue"],
            labels={"accuracy": "Accuracy", "description": feature_name},
            height=500,
        )

        fig.update_traces(
            textposition="inside", textfont=dict(color="white", size=14, family="Arial")
        )

        fig.update_layout(
            title=f"{feature_name} - Subgroup Accuracy Histogram",
            yaxis=dict(range=[0, 1]),
            uniformtext_minsize=12,
            uniformtext_mode="hide",
        )

        fig.show()
        fig.write_image(f"C:\\code\\swe-bench\\main\\results\\current.pdf")


agent_name1 = "20251120_livesweagent_gemini-3-pro-preview"
agent_name2 = "20251103_sonar-foundation-agent_claude-sonnet-4-5"

HistogramsForQuantiles(BenchmarkType.VERIFIED, agent_name1).cli()

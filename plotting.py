import plotly.graph_objects as go


class PlottingManager:
    def __init__(self, df):
        self.input_df = df

    def retrieve_agent_names(self):
        agent_names = [
            col.strip("resolve_status_")[1]
            for col in self.input_df.columns
            if col.startswith("resolve_status_")
        ]
        return agent_names

    # def plot(self, feature_name):
    #
    #     # Create base figure
    #     fig = go.Figure()
    #
    #     # --- Base histogram: all data (always visible)
    #     fig.add_trace(
    #         go.Histogram(
    #             x=self.input_df[feature_name],
    #             # nbinsx=20,
    #             name="All data",
    #             opacity=0.4,
    #             marker=dict(color="gray", line=dict(color="black", width=1)),
    #             visible=True,
    #         )
    #     )
    #
    #     # --- Add one histogram per condition column (initially only first visible)
    #     condition_cols = [
    #         col_name
    #         for col_name in self.input_df.columns
    #         if col_name.startswith("binary_resolve_status_")
    #     ]
    #     for i, cond in enumerate(condition_cols):
    #         fig.add_trace(
    #             go.Histogram(
    #                 x=self.input_df.loc[self.input_df[cond], feature_name],
    #                 # nbinsx=20,
    #                 name=f"{cond} = True",
    #                 opacity=0.75,
    #                 marker=dict(color="blue", line=dict(color="black", width=1)),
    #                 visible=(i == 0),  # show only first condition initially
    #             )
    #         )
    #
    #     # --- Create dropdown menu to toggle condition histograms
    #     buttons = []
    #     for i, cond in enumerate(condition_cols):
    #         visible = [True] + [j == i for j in range(len(condition_cols))]
    #         buttons.append(
    #             dict(
    #                 label=cond,
    #                 method="update",
    #                 args=[
    #                     {"visible": visible},
    #                     {"title": f"Histogram Comparison (All data vs {cond}=True)"},
    #                 ],
    #             )
    #         )
    #
    #     # --- Add dropdown to layout
    #     fig.update_layout(
    #         updatemenus=[
    #             {
    #                 "buttons": buttons,
    #                 "direction": "down",
    #                 "showactive": True,
    #                 "x": 0.5,
    #                 "xanchor": "center",
    #                 "y": 1.15,
    #                 "yanchor": "top",
    #             }
    #         ],
    #         barmode="overlay",
    #         title="Histogram Comparison ",
    #         xaxis_title=feature_name.replace("_", " ").title(),
    #         yaxis_title="Frequency",
    #         legend_title="Data subsets",
    #     )
    #
    #     return fig

    def plot(self, feature_name):

        # Calculate bin edges based on ALL data (or just the main range you care about)
        # Option 1: Use all data
        # x_data = self.input_df[feature_name].dropna()

        # Option 2: Focus on main range (0-10) and ignore outliers for binning
        x_data = self.input_df[feature_name].dropna()
        x_data_filtered = x_data[x_data <= 10]  # Adjust threshold as needed

        # Create consistent bins
        x_min = x_data_filtered.min()
        x_max = x_data_filtered.max()
        nbins = 20
        bin_size = (x_max - x_min) / nbins

        # Create base figure
        fig = go.Figure()

        # --- Base histogram: all data (always visible)
        fig.add_trace(
            go.Histogram(
                x=self.input_df[feature_name],
                xbins=dict(start=x_min, end=x_max, size=bin_size),
                name="All data",
                opacity=0.4,
                marker=dict(color="gray", line=dict(color="black", width=1)),
                visible=True,
            )
        )

        # --- Add one histogram per condition column (initially only first visible)
        condition_cols = [
            col_name
            for col_name in self.input_df.columns
            if col_name.startswith("binary_resolve_status_")
        ]
        for i, cond in enumerate(condition_cols):
            fig.add_trace(
                go.Histogram(
                    x=self.input_df.loc[self.input_df[cond], feature_name],
                    xbins=dict(start=x_min, end=x_max, size=bin_size),
                    name=f"{cond} = True",
                    opacity=0.75,
                    marker=dict(color="blue", line=dict(color="black", width=1)),
                    visible=(i == 0),  # show only first condition initially
                )
            )

        # --- Create dropdown menu to toggle condition histograms
        buttons = []
        for i, cond in enumerate(condition_cols):
            visible = [True] + [j == i for j in range(len(condition_cols))]
            buttons.append(
                dict(
                    label=cond,
                    method="update",
                    args=[
                        {"visible": visible},
                        {"title": f"Histogram Comparison (All data vs {cond}=True)"},
                    ],
                )
            )

        # --- Add dropdown to layout
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.15,
                    "yanchor": "top",
                }
            ],
            barmode="overlay",
            title="Histogram Comparison ",
            xaxis_title=feature_name.replace("_", " ").title(),
            yaxis_title="Number of instances",
            legend_title="Data subsets",
            xaxis=dict(
                range=[x_min, x_max * 1.05]
            ),  # Set x-axis range to focus on main data
        )

        return fig

# import plotly.graph_objects as go
# import pandas as pd
# import numpy as np
#
# # Example dataset
# np.random.seed(0)
# df = pd.DataFrame(
#     {
#         "values": np.random.normal(50, 10, 200),
#         "condition1": np.random.choice([True, False], 200),
#         "condition2": np.random.choice([True, False], 200),
#         "condition3": np.random.choice([True, False], 200),
#     }
# )
#
# # Define the condition columns you want to switch between
# condition_columns = ["condition1", "condition2", "condition3"]
#
# # --- Base figure ---
# fig = go.Figure()
#
# # Histogram for all data (background)
# fig.add_trace(
#     go.Histogram(
#         x=df["values"],
#         nbinsx=20,
#         name="All data",
#         opacity=0.4,
#         marker=dict(color="gray", line=dict(color="black", width=1)),
#     )
# )
#
# # Foreground histogram — start with first condition
# fig.add_trace(
#     go.Histogram(
#         x=df.loc[df[condition_columns[0]], "values"],
#         nbinsx=20,
#         name=f"{condition_columns[0]} = True",
#         opacity=0.75,
#         marker=dict(color="blue", line=dict(color="black", width=1)),
#     )
# )
#
# # --- Create dropdown menu ---
# dropdown_buttons = []
#
# for cond in condition_columns:
#     # Each button updates the second trace (index 1)
#     dropdown_buttons.append(
#         dict(
#             label=cond,
#             method="update",
#             args=[
#                 {
#                     "x": [df["values"], df.loc[df[cond], "values"]],
#                     "name": ["All data", f"{cond} = True"],
#                 },
#                 {"title": f"Histogram Comparison (All data vs {cond}=True)"},
#             ],
#         )
#     )
#
# # --- Layout ---
# fig.update_layout(
#     barmode="overlay",
#     title=f"Histogram Comparison (All data vs {condition_columns[0]}=True)",
#     xaxis_title="Values",
#     yaxis_title="Frequency",
#     legend_title="Data subsets",
#     updatemenus=[
#         {
#             "buttons": dropdown_buttons,
#             "direction": "down",
#             "showactive": True,
#             "x": 1.05,
#             "xanchor": "left",
#             "y": 1.05,
#             "yanchor": "top",
#         }
#     ],
# )
#
# fig.show()

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Example dataset
np.random.seed(0)
df = pd.DataFrame(
    {
        "values": np.random.normal(50, 10, 200),
        "condition1": np.random.choice([True, False], 200),
        "condition2": np.random.choice([True, False], 200),
        "condition3": np.random.choice([True, False], 200),
    }
)

# Create base figure
fig = go.Figure()

# --- Base histogram: all data (always visible)
fig.add_trace(
    go.Histogram(
        x=df["values"],
        nbinsx=20,
        name="All data",
        opacity=0.4,
        marker=dict(color="gray", line=dict(color="black", width=1)),
        visible=True,
    )
)

# --- Add one histogram per condition column (initially only first visible)
condition_cols = ["condition1", "condition2", "condition3"]
for i, cond in enumerate(condition_cols):
    fig.add_trace(
        go.Histogram(
            x=df.loc[df[cond], "values"],
            nbinsx=20,
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
    title="Histogram Comparison (All data vs condition1=True)",
    xaxis_title="Values",
    yaxis_title="Frequency",
    legend_title="Data subsets",
)

fig.show()

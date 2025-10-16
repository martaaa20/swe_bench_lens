import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Example dataset
np.random.seed(0)
df = pd.DataFrame(
    {
        "values": np.random.normal(50, 10, 200),
        "condition1": np.random.choice([True, False], 200),
    }
)

# Create figure
fig = go.Figure()

# Background: all data
fig.add_trace(
    go.Histogram(
        x=df["values"],
        nbinsx=20,
        name="All data",
        opacity=0.4,
        marker=dict(
            color="gray",
            line=dict(color="black", width=1),  # border color  # border width (in px)
        ),
    )
)

# Foreground: only condition1 == True
fig.add_trace(
    go.Histogram(
        x=df.loc[df["condition1"], "values"],
        nbinsx=20,
        name="Condition1 = True",
        opacity=0.5,
        marker=dict(
            color=px.colors.qualitative.Vivid[3],
            line=dict(color="green", width=1),  # border color  # border width (in px)
        ),
    )
)

# Layout
fig.update_layout(
    barmode="overlay",  # overlay instead of stacking
    title="Histogram Comparison (All data vs condition1=True)",
    xaxis_title="Values",
    yaxis_title="Frequency",
    legend_title="Data subsets",
)

fig.show()

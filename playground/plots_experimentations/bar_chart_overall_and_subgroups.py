"""This file contains the plot (bar chart) that firstly shows general score, and then the 2 subgroups"""

import plotly.express as px
import plotly.graph_objects as go

df1 = px.data.tips()

# Calculate overall average by sex
overall_avg = df1.groupby("sex")["total_bill"].mean().reset_index()

# Calculate average by sex and smoker
smoker_avg = df1.groupby(["sex", "smoker"])["total_bill"].mean().reset_index()

# Create figure
fig1 = go.Figure()

# First layer: Overall average (one color)
fig1.add_trace(
    go.Bar(
        x=overall_avg["sex"],
        y=overall_avg["total_bill"],
        name="Overall Average",
        marker_color="black",
        opacity=0.6,
        offsetgroup="0",  # Add this to group it with others
    )
)

# Second layer: Breakdown by smoker status
for i, smoker_status in enumerate(smoker_avg["smoker"].unique(), start=1):
    data = smoker_avg[smoker_avg["smoker"] == smoker_status]
    fig1.add_trace(
        go.Bar(
            x=data["sex"],
            y=data["total_bill"],
            name=f"Smoker: {smoker_status}",
            offsetgroup=str(i),  # Use sequential numbers
        )
    )

fig1.update_layout(
    barmode="group",  # Remove the extra space in "group "
    height=400,
    xaxis_title="Sex",
    yaxis_title="Average Total Bill",
)

fig1.show()

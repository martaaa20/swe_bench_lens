import pysubgroup as ps
import pandas as pd

# Load the example dataset
from pysubgroup.datasets import get_titanic_data

data = get_titanic_data()

# Define what we want to analyze (target)
target = ps.BinaryTarget("Survived", True)

# Create search space (all possible conditions except the target)
searchspace = ps.create_selectors(data, ignore=["Survived"])

# Define the search task
task = ps.SubgroupDiscoveryTask(
    data,
    target,
    searchspace,
    result_set_size=5,  # Find top 5 subgroups
    depth=2,  # Maximum 2 conditions per subgroup
    qf=ps.WRAccQF(),  # Use Weighted Relative Accuracy measure
)

# Execute the search
result = ps.DFS().execute(task)

# Display results
print(result.to_dataframe())

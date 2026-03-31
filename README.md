
# Set up the framework:
#### 1. execute the `benchmark_downloader.py` script to execute all the benchmarks from HuggingFace.
#### 2. pull the repository experiments with the evaluation results of agents: https://github.com/SWE-bench/experiments
#### 3. Create Folders

1. Create a `/data` folder at the root of this repository.  
2. Inside `/data`, create subfolders for the benchmark results.  
   - The name of each subfolder should match the corresponding value in the `BenchmarkType` enumerator.  
3. Copy the agent results from the `experiments` repository into the appropriate benchmark subfolder.  

At the end, the directory structure should look like this:

```text
swe-bench/
└── data/
    ├── swe-bench-verified/
    │   ├── 20260208_Agent_Example_1/
    │   │   ├── results/
    │   │   │   ├── resolved_by_repo.json
    │   │   │   ├── resolved_by_time.json
    │   │   │   └── results.json
    │   │   ├── metadata.yaml
    │   │   └── README.md
    │   └── 20260209_Agent_Example_2/
    │       └── ...
    └── swe-bench-lite/
        └── ...
```
  




# Framework execution:
There are the following points how one can execute the framework:
1) **subgroup analysis pipeline** 
   - GOAL: execute the subgroup analysis pipeline to discover the subgroups of an agent
   - `main/subgroup_analysis/subgroup_analysis_pipeline.py`

2) **comparison of agents** 
   - GOAL: execute the subgroup analysis pipeline AND compare how other agents perform compared to that agent by looking at their performance and its changes
   - unlimited amount of agents
   - `main/results/compare_agents.py`

3) **agent profiler (final analysis framework)**
   - GOAL: get a report on statistics based on discovered subgroups
   - this can be used for exploratory data analysis
   - better understanding of the subgroups and results with this framework
   - `main/results/agent_profiler.py`

4) **plotting the data for an agent based on features**
   - GOAL: visual analysis of the features of the agent
   - better interpretability of the data
   - exploration is encouraged
   - `main/results/histograms_for_quantiles.py`

## Define the GitHub token
Create an .env file in the project’s root directory. Put the GitHub token there as follows:
GITHUB_TOKEN=your_github_api_token
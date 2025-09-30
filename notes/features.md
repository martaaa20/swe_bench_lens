# First steps - Bachelor thesis

## Features of the SWE-bench

### Repo characteristics

#### Features:
- programming language
- code lines
- number of contributors

#### Discussion
These features would indicate what kind of environment is the agent good at.
Example outcomes:
- Agent X works best on small python repositories.
- The best agent for big java projects (evaluated by number of contributors and code lines) is agent X


### Issue characteristics
#### Features
- number of FAIL_TO_PASS
- files hierarchy (what is the deepest delta of file location between files): shows how much of the repository an agent would have to consider to come to the solution
- number of files changed
- hunk count
- patch spread: improvement idea - relative to code lines in the file. if it has to be modified at 6 places of a 1200 code lines, it's equivalent to being modified once in 200 lines
- number of additions
- number of deletions (remark: consider renamings in golden truth patches -> not necessary to solve the problems)
- lines_of_delta_total
- programming language of the solution / file extension of the files affected (relative to the amount of changes: code lines / percentage of all the changes)

## Issue quality
### Features
- typos in the issue descriptions
- length of description
- errors copy/pasted
- code mentions: extracted identifiers
- chain-of-thought words ((L)LM-judged / programmatically extractable: by occurences of words, like "firstly", "leads to" etc. (?))
- clues on how to solve ((L)LM-judged?)
- understandability of description ((L)LM-judged?)



## Problems:
 
### Code maintainability of AI-written code
Code maintance after AI models writing code, becomes bigger and bigger of a problem. 
For this, the agents should have the workflow of code maintanance after assuming to have solved the problem and all tests running successfully.
Such a thing should also be considered in the score of an agent, maybe an additional one, that shows how well an agent can modify code without making it hard to maintain.
The score on the repo before and after applying changes, only mentioned to the methods / classes changed by the patch.

Tools for scoring the code maintanability:
- static analysis - SonarQube
  - The commercial of.ferings of SonarQube supports programming languages such as Java (including Android), C#, C, C++, JavaScript, TypeScript, Python, Go, Swift, COBOL, Apex, PHP, Kotlin, Ruby, Scala, HTML, CSS, ABAP, Flex, Objective-C, PL/I, PL/SQL, RPG, T-SQL, VB.NET, VB6, and XML.
  - Calculates a Maintainability Index (MI) from 0 to 100, where higher scores indicate easier-to-maintain code. It tracks code smells, technical debt, and violation of coding standards, integrating the score into its dashboard and automatic reports


# First steps
1. create distributions on the issues features
2. create a score for issue quality: SW````E-bench verified already solves that problem: but to which extent. could be a valuable indicator for SWE-smith issues
3. run for different agents and compare their scores on based on those features

Next steps:
## Phase 1: Dataset building
1. Build a feature extraction pipeline: with caching and parallel implementations 
   - question: which benchmark (SWE-bench Verified / SWE-bench / Lite / refactorbench(?) / swe bench pro)
   - PR arena: in the wilderness
   - comparison on swe-bench verified and pr arena
   - 
2. Dataset building: verification of the correct feature extraction (e.g. verification of LLM answers)
3. Feature correlation tool: identification of redundant or highly correlated features
4. Feature distributions across different repositories/programming languages (repository features)
5. Creating visualizations
plotting 

## Phase 2: Subgroup analysis
Help from Jurgen n ot a problem
Finding subgroups that have specific intresting behaviour.
Evaluation of results for different agents, e.g. openAI codex, GitHub Copilot coding agent, agents running on smaller models to find the fastest/most efficient for specific issues.
Concern: as the AI models get better and better, the results for the most STOA agents, can be quite similar. 

1. get data on some different agents performances
2. implement subgroup discovery algorithm:
   - subgroup coverage, effect size, statistical significance
3. (maybe) framework for defining "interestingness" measures
   - is it randomness that makes the subgroup have different results, or can we actually say this subgroup is interesting
4. implement subgroup mining: tree-based methods, apriori algorithm, shap, clustering approaches(?)

## Phase 3: agent-specific analysis
Find where the agents differ most. 

1. apply the feature subgroups on agents results.
2. discuss their differences: dashboards, statistical analysis

## Phase 4: predictive modeling
Build statistical models to perform routing to agents, resulting in a multi-agent framework






# Technologies / frameworks used for the project
- pandas / numpy for small dataset
- dask for bigger dataset:  good alternative for pandas but more data
- conda 

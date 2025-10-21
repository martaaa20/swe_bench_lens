Step 1:
show the things I've implemented up until now
- concentrated on: Verified SWE bench

 - Benchmark and agents results downloaders
 - merge the benchmark and agent results if it could resolver the issues

 - adding features class : AddFeaturesPipeline
 - fixing dataset: patches had typos: mostly wiht the newlines (fixed 27 manually)
 - used the library unidiff for extracting the feature values for each instance 
 - then started a bit with visualizations (used AI for creating some templates)

Visualizations:
- correlation matrix (from 0.7-0.8 on) none have that strong of relationship
- agents vs feature
- for outliers (they make graphs unreadable, because they influence deciusion on the amount of bins when creating histograms)
    - could use the data only that is in 2-3 standard derivations
    - identify the hot zone and show that

- no clue if that idea of visualization is good enough to notice some interesting points (maybe to notice some thign and look them up more detailed later on)



For comparing the different benchmarks:
- comparing histograms, bar charts
- comparing using the visualizations similar to the one from the Google paper (ECDF)
- QUESTION: how do we put into numbers how different benchmarks differ from each other?
  - by analyzing the plots? - getting numbers from those

For comparing the different agents performances:
- QUESTION: how do we put into numbers how different benchmarks differ from each other?
  - finding some kind of correlation between the features of how similar the instances are that are solved by an agent1 and agent2 based on different features



Next steps:
1. Feature distributions across different repositories(programming languages) (repository features)
   - so wokring with other benchamrks: swe bench lite, pr arena etc.
2. visualize them
QUESTION: what is the first goal: finding those differences between benchmarks rather than agents results, right?


Organizational:
- when to start writing optimally
- is there a limitation on amount of pages/ definition of how much work needs to be done exactly


## Notes after meeting
### TODOs:
- github
  - add prof to the github repo
    - his username: citostyle
  - at the end will be published to https://github.com/ipa-lab/
  - repo can be private or public: does not matter
  - add the data to the repo so you can easily run it

- mit dem Schreiben von Bachelorarbeit anfangen
  - die Links fur die Hinweisen wie man das macht hab ich bekommen
  -  keine evaluation, approach
  - anstatt von dem introduction, background, related work, forschungsfragen
  - die Introduction schon schreiben um die Ziele zu identifizieren
  - background: what is subgroup analysis and what is SWE bench
  - related work: ??
  - forschungsfragen: was interessiert mich mit dem
  - TU bachelorarbeit template bekommen

- writing: organisational:
  - so wenig wie moglich schreiben
  - am Ende 30-40 Seiten haben die Leute meistens
  - hinweisen on how to write:
    - https://docs.google.com/document/d/1csIMGsvTcleUnpkGxn6qaW9LEs3QlmosLcfo1HHGo-I/edit?tab=t.0#heading=h.qzfsxd98cefi
    - https://docs.google.com/document/d/1rgIigzeHZbZrrvxfghyF2flsigdJITyu9rGPUCq2a_E/edit?tab=t.0#heading=h.te60tlkpkqr
    - 

- next programming steps:
  - ziel for now: mvp end-to-end with verified
  - Example: 70 % -> subgroup analysis: num_of_hunks > 2 und num_of_files_hierarchy -> performance drop auf 20%
  - subgroup analysis: 
    - https://pypi.org/project/subgroups/
    - https://pysubgroup.readthedocs.io/en/latest/

### Next meetings:
- Let professor know, when I've written stuff down
- on the 30th meeting with Jatin
  - show the features 
  - show the repo / script what I've done up until now

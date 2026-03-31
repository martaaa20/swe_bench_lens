from data_structures.benchmark_type_enum import BenchmarkType
from main.results.paper_writing_helpers.rewrite_to_latex import (
    rewrite_interval_expression2,
)
from main.subgroup_analysis.subgroup_analysis_pipeline import SubgroupAnalysisPipeline

if __name__ == "__main__":
    result = ""
    agent_name1 = "20251120_livesweagent_gemini-3-pro-preview"
    agent_name2 = "20251103_sonar-foundation-agent_claude-sonnet-4-5"
    sg_disc2 = SubgroupAnalysisPipeline(agent_name2, BenchmarkType.VERIFIED, False)
    subgroups2 = sg_disc2.perform()
    sg_list = sg_disc2.get_data(subgroups2)
    for i, sg_tuple in enumerate(sg_list):
        subgroup_description_latex = rewrite_interval_expression2(sg_tuple[0])
        num_instances = sg_tuple[1]
        sg_acc = sg_tuple[2]
        delta_acc = sg_tuple[3]

        sign = "" if str(delta_acc).startswith("-") else "+"
        latex_row_str = f"{i+67} & {subgroup_description_latex}  & {num_instances} & {f"{sign}{delta_acc*100}"[:5]}\\% & {str(sg_acc*100)[:4]}\\% \\\\\n\\hline\n"
        result += latex_row_str
    print(result)

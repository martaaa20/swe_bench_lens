from enum import Enum

from pydantic import BaseModel


class CategoryOfFeature(Enum):
    REPOSITORY = "Repository-specific features"
    ISSUE_DESCRIPTION = "Issue-description features"
    GROUND_TRUTH = "Issues' ground truth features"


class Subgroup(BaseModel):
    selector_str: str
    selectors: tuple
    num_instances: int
    subgroup_accuracy: float
    subgroup_interestingness: float


class SubgroupAnalysisResultModel(BaseModel):
    subgroups: list[Subgroup]
    overall_accuracy: float
    overall_instances_count: int

    def pretty_print(self):
        for idx, subgroup_instance in enumerate(self.subgroups):
            print(f"\n\n######## SUBGROUP {idx + 1}")
            print(subgroup_instance.selector_str)
            print("score: " + str(subgroup_instance.subgroup_interestingness))
            print(
                f"number of instances of the subgroup: {subgroup_instance.num_instances}"
            )
            print(f"accuracy in the SUBGROUP: {subgroup_instance.subgroup_accuracy}")
            print(f"VS accuracy in the full dataset: {self.overall_accuracy}")

    def get_data(self) -> list[tuple]:

        subgroups_info = []
        for sg in self.subgroups:
            delta_acc = sg.subgroup_accuracy - self.overall_accuracy
            sg_tuple = (
                sg.selector_str,
                sg.num_instances,
                sg.subgroup_accuracy,
                delta_acc,
            )
            subgroups_info.append(sg_tuple)

        return subgroups_info

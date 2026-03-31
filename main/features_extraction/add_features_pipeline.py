import ast
import codecs
import collections
import itertools
import pickle
import re
from pathlib import Path

import pandas as pd
from unidiff.patch import PatchSet

from main.features_extraction.fixed_patches import fixed_patches_dict
from main.features_extraction.github_scraping import GitHubInfo


class AddFeaturesPipeline:

    def __init__(self, input_df):
        self.input_df = input_df.copy()
        self.input_df = self._add_fixed_patches(self.input_df)
        self.output_df = None

    def get_original_df_with_features(self) -> pd.DataFrame:

        if self.output_df is None:
            self.execute()
        return self.output_df

    def execute(self):
        if self.output_df is not None:
            return

        # features commented out were not used in the final version, but are good for exploratory data analysis
        executables = [
            # self._add_num_of_fail_to_pass,
            # self._add_num_of_pass_to_pass,
            self._add_num_of_hunks,
            self._add_num_of_files_changed,
            self._add_files_hierarchy_delta,
            self._add_patch_spread,
            self._add_num_of_modified_lines,
            # self._add_num_of_deletions,
            # self._add_num_of_additions,
            # self._add_delta_of_new_lines,
            self._add_length_of_description,
            self._add_num_of_code_mentions,
            self._add_repository_name_feature,
            self._add_difficulty_binary_features,
        ]
        result = self.input_df
        for add_feature_function in executables:
            if "difficulty" in result.columns:
                pass
            else:
                pass
            result = add_feature_function(result)
        self.output_df = result

    def get_df_for_correlation(self):
        if self.output_df is None:
            self.execute()

        column_names_feats = [
            col for col in self.output_df.columns if col.startswith("FEAT_")
        ]
        only_features_df = self.output_df.set_index("instance_id")[column_names_feats]
        return only_features_df

    def get_df_for_subgroup_analysis(self):
        # the only difference is here one-hot encoded difficulty column is not needed
        if self.output_df is None:
            self.execute()

        # get the repository-level statistics
        repo_features_df = self.get_repo_features()
        repo_features_df.columns = [
            col if (col == "repo") else f"FEAT_{col}"
            for col in repo_features_df.columns
        ]

        # the following line adds the repository features
        #  commented out, because those weren't used in the final version
        # final_df = self.output_df.merge(repo_features_df, on="repo")
        final_df = self.output_df

        column_names_feats = [
            col
            for col in final_df.columns
            if col.startswith("FEAT_") and not col.startswith("FEAT_difficulty")
        ]

        column_names_feats.extend(["binary_resolved"])
        result_df = final_df.set_index("instance_id")[column_names_feats]
        return result_df

    def get_corr_matrix(self):
        features_df = self.get_df_for_correlation()
        return features_df.corr()

    def get_repo_features(self):

        my_file = Path(
            "C:/code/swe-bench/main/features_extraction/swe_bench_verified_repo_stats.pickle"
        )
        if my_file.is_file():
            with open(my_file, "rb") as f:
                df = pickle.load(open(my_file, "rb"))

            return df

        repo_names = set(list(self.input_df["repo"]))

        repo_values_list = []
        for repo_name in repo_names:
            github_info = GitHubInfo(repo_name)
            repo_values = github_info.get_all_features_dict()
            repo_values_list.append(repo_values.values())

        column_names = list(repo_values.keys())
        df = pd.DataFrame(data=repo_values_list, columns=column_names)
        return df

    @staticmethod
    def _add_fixed_patches(df):
        df["patch_fixed"] = pd.NA
        for index, row in df.iterrows():
            if row["instance_id"] in fixed_patches_dict.keys():
                df.at[index, "patch_fixed"] = fixed_patches_dict[row["instance_id"]]
            else:
                df.at[index, "patch_fixed"] = codecs.decode(
                    row["patch"], "unicode_escape"
                )
        return df

    @staticmethod
    def _add_num_of_fail_to_pass(input_df):
        input_df["FEAT_num_of_fail_to_pass"] = (
            input_df["FAIL_TO_PASS"].apply(ast.literal_eval).apply(len)
        )
        return input_df

    @staticmethod
    def _add_num_of_pass_to_pass(input_df):
        input_df["FEAT_num_of_pass_to_pass"] = (
            input_df["PASS_TO_PASS"].apply(ast.literal_eval).apply(len)
        )
        return input_df

    @staticmethod
    def _add_num_of_hunks(input_df):
        def get_num_of_hunks(row):
            hunks_num = 0
            patch_set = PatchSet.from_string(row["patch_fixed"])
            for patched_file in patch_set:
                hunks_num += len(patched_file)

            return hunks_num

        input_df["FEAT_num_of_hunks"] = input_df.apply(get_num_of_hunks, axis=1)
        return input_df

    @staticmethod
    def _add_num_of_files_changed(input_df):
        input_df["FEAT_num_of_files_changed"] = input_df.apply(
            lambda x: len(PatchSet.from_string(x["patch_fixed"])), axis=1
        )
        return input_df

    @staticmethod
    def _add_files_hierarchy_delta(input_df):
        """
        Calculates and adds top the `input_df` a column with an int of files hierarchy delta.
        Example: a/b/c/utils/whatever/d.py
                 a/b/c/other/d.py
                 result = 5 (d.py -> whatever -> utils -> c -> other -> d.py)
        """

        def get_prefix_of_strings(string1, string2):
            filenames = [string1, string2]
            prefix = filenames[0]
            for s in filenames[1:]:
                while not s.startswith(prefix):
                    prefix = "/".join(prefix.split("/")[:-1])
                    if not prefix:
                        return ""
                prefix = prefix + "/"
            return prefix

        def get_hops(row):
            patched_set = PatchSet.from_string(row["patch_fixed"])
            patched_filenames = [patched_file.path for patched_file in patched_set]
            if len(patched_filenames) == 1:
                return 0

            max_hops = 0
            for str1, str2 in itertools.combinations(patched_filenames, 2):
                prefix = get_prefix_of_strings(str1, str2)
                hops = sum(
                    [len(elem.strip(prefix).split("/")) for elem in [str1, str2]]
                )
                if hops > max_hops:
                    max_hops = hops
            return hops

        input_df["FEAT_files_hierarchy_delta"] = input_df.apply(get_hops, axis=1)
        return input_df

    @staticmethod
    def _add_patch_spread(input_df):
        def get_patch_spread(row):
            patch_set = PatchSet.from_string(row["patch_fixed"])
            patch_spread = 0
            for patched_file in patch_set:
                if len(patched_file) > 1:
                    for i in range(len(patched_file) - 2):
                        hunk_first = patched_file[i]
                        hunk_last = patched_file[i + 1]
                        patch_spread_hunks = hunk_last.source_start - (
                            hunk_first.source_start + hunk_first.source_length
                        )
                        patch_spread += patch_spread_hunks
            return patch_spread

        input_df["FEAT_patch_spread"] = input_df.apply(get_patch_spread, axis=1)
        return input_df

    @staticmethod
    def _add_num_of_modified_lines(input_df):
        """
        This is specific to the SWE benchmarks.
        """

        def get_num_of_modified_lines(row):
            num_modified = 0
            patch_set = PatchSet.from_string(row["patch_fixed"])

            for patched_file in patch_set:
                num_modified += patched_file.added
                num_modified += patched_file.removed
            return num_modified

        input_df["FEAT_num_of_modified_lines"] = input_df.apply(
            get_num_of_modified_lines, axis=1
        )
        return input_df

    @staticmethod
    def _add_num_of_deletions(input_df):
        """
        This is not used for SWE benchmarks. Instead, modified lines are used
        """

        def get_num_of_deletions(row):
            num_deletions = 0
            patch_set = PatchSet.from_string(row["patch_fixed"])

            for patched_file in patch_set:
                num_deletions += patched_file.removed
            return num_deletions

        input_df["FEAT_num_of_deletions"] = input_df.apply(get_num_of_deletions, axis=1)
        return input_df

    @staticmethod
    def _add_num_of_additions(input_df):
        """
        This is not used for SWE benchmarks. Instead, modified lines are used
        """

        def get_num_of_additions(row):
            num_additions = 0
            patch_set = PatchSet.from_string(row["patch_fixed"])

            for patched_file in patch_set:
                num_additions += patched_file.added
            return num_additions

        input_df["FEAT_num_of_additions"] = input_df.apply(get_num_of_additions, axis=1)
        return input_df

    @staticmethod
    def _add_delta_of_new_lines(input_df):

        def get_delta_of_new_lines(row):
            delta = 0
            patch_set = PatchSet.from_string(row["patch_fixed"])

            for patched_file in patch_set:
                delta += patched_file.added
                delta -= patched_file.removed
            return delta

        input_df["FEAT_num_of_additions"] = input_df.apply(
            get_delta_of_new_lines, axis=1
        )
        return input_df

    @staticmethod
    def _add_length_of_description(input_df):
        input_df["FEAT_length_of_description"] = input_df.apply(
            lambda x: len(x["problem_statement"]), axis=1
        )
        return input_df

    @staticmethod
    def _add_num_of_code_mentions(input_df):
        def get_num_of_code_mentions(row):
            snake_case_pattern = r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b"
            camel_case_pattern = r"\b[A-Z][a-z0-9]*[A-Z][a-zA-Z0-9]*\b"

            text = row["problem_statement"]
            snake_matches = re.findall(snake_case_pattern, text)
            camel_matches = re.findall(camel_case_pattern, text)
            all_matches = snake_matches + camel_matches

            return len(all_matches)

        input_df["FEAT_num_of_code_mentions"] = input_df.apply(
            get_num_of_code_mentions, axis=1
        )
        return input_df

    @staticmethod
    def _add_difficulty_binary_features(input_df):
        df_copy = input_df.copy()
        result = pd.get_dummies(
            df_copy, columns=["difficulty"], prefix="FEAT_difficulty"
        )
        result["difficulty"] = df_copy[
            "difficulty"
        ]  # because pd.get_dummies deletes the original column
        return result

    @staticmethod
    def _add_repository_name_feature(input_df):
        input_df["FEAT_repository_name"] = input_df["repo"]
        return input_df

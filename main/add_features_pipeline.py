import ast
import codecs
import collections
import itertools
import re

import pandas as pd
from unidiff.patch import PatchSet

from main.fixed_patches import fixed_patches_dict


class AddFeaturesPipeline:

    def __init__(self, input_df):
        self.input_df = input_df.copy()
        self.input_df = self._add_fixed_patches(self.input_df)
        self.output_df = None

    def get_original_df_with_features(self) -> pd.DataFrame:
        if self.output_df is not None:
            return self.output_df
        executables = [
            self._add_num_of_fail_to_pass,
            self._add_num_of_pass_to_pass,
            self._add_num_of_hunks,
            self._add_num_of_files_changed,
            self._add_files_hierarchy_delta,
            self._add_patch_spread,
            self._add_num_of_deletions,
            self._add_num_of_additions,
            self._add_delta_of_new_lines,
            self._add_length_of_description,
            self._add_num_of_code_mentions,
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

        return self.output_df

    def get_only_features_df(self):
        if self.output_df is None:
            self.execute()

        column_names_feats = [
            col for col in self.output_df.columns if col.startswith("FEAT_")
        ]
        only_features_df = self.output_df.set_index("instance_id")[column_names_feats]
        return only_features_df

    def get_corr_matrix(self):
        features_df = self.get_only_features_df()
        return features_df.corr()

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
        # TODO: check if this makes sense
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
    def _add_num_of_deletions(input_df):
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
        print(input_df.columns)
        df_copy = input_df.copy()
        result = pd.get_dummies(
            df_copy, columns=["difficulty"], prefix="FEAT_difficulty"
        )
        result["difficulty"] = df_copy[
            "difficulty"
        ]  # because pd.get_dummies deletes the original column
        return result

    # ---------- not used/implemented features -------------------------------------------------------------------------
    @staticmethod
    def _add_programming_language_percentages(input_df):
        """
        DO NOT USE THIS
        - does not make sense for SWE-bench verified
        - implementation not finished

        This function does not make sense: Extensions from all issues {'cfg': 1, 'py': 622}
        (for SWE-Bench verified)
        """
        extensions = []
        for index, row in input_df.iterrows():
            patch_set = PatchSet.from_string(row["patch_fixed"])
            extensions.extend([elem.path.split(".")[-1] for elem in patch_set])

        counter = collections.Counter(extensions)

        return input_df

    @staticmethod
    def _add_relative_patch_spread(input_df):
        # TODO: implement
        #   for future, currently too much effort to extract num of lines of code of each file
        return input_df

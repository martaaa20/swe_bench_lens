import re


def rewrite_interval_expression(expr: str) -> str:
    """
    Rewrites interval expressions of the form:
    name: [a:b[ AND name2: [c:d[
    into:
    name ∈ [a, b) ∧ name2 ∈ [c, d)
    """
    # rewrite the slots and the "\in"
    pattern = r"(\w+)\s*:\s*\[(\d+)\s*:\s*(\d+)\["
    matches = re.findall(pattern, expr)
    rewritten_parts = [f"{name} \\in [{start}, {end})" for name, start, end in matches]
    temp_result = " \\land ".join(rewritten_parts)

    # map the feature names to shorter and more understandable versions
    mapping_feat_names = {
        "FEAT_num_of_hunks": "num_hunks",
        "FEAT_num_of_files_changed": "num_files_changed",
        "FEAT_files_hierarchy_delta": "files_hierarchy_delta",
        "FEAT_patch_spread": "patch_spread",
        "FEAT_num_of_modified_lines": "num_modified_lines",
        "FEAT_length_of_description": "description_length",
        "FEAT_num_of_code_mentions": "num_code_mentions",
        "FEAT_repository_name": "repository_name",
    }
    for feat_name, change_to_name in mapping_feat_names.items():
        temp_result = temp_result.replace(feat_name, "\\text{" + change_to_name + "}")

    # add to each underline sign the backslash to it (latex needs it)
    temp_result = temp_result.replace("_", "\\_")

    result = f"${temp_result}$"
    return result


import re


def rewrite_interval_expression2(expr: str) -> str:
    """
    Rewrites interval expressions like:
    FEAT_length_of_description: [1187:2039[ AND FEAT_num_of_modified_lines<3
    into:
    \text{description_length} ∈ [1187, 2039) ∧ \text{num_modified_lines}<3
    """
    # Map the feature names first
    mapping_feat_names = {
        "FEAT_num_of_hunks": "num_hunks",
        "FEAT_num_of_files_changed": "num_files_changed",
        "FEAT_files_hierarchy_delta": "files_hierarchy_delta",
        "FEAT_patch_spread": "patch_spread",
        "FEAT_num_of_modified_lines": "num_modified_lines",
        "FEAT_length_of_description": "description_length",
        "FEAT_num_of_code_mentions": "num_code_mentions",
        "FEAT_repository_name": "repository_name",
    }

    # Replace feature names with \text{...}
    for feat_name, change_to_name in mapping_feat_names.items():
        expr = expr.replace(feat_name, f"\\text{{{change_to_name}}}")

    # Replace underscores with \_ for LaTeX
    expr = expr.replace("_", "\\_")

    # Rewrite interval expressions [a:b[ into [a, b)
    pattern = r"(\\text\{[^\}]+\})\s*:\s*\[(\d+)\s*:\s*(\d+)\["
    expr = re.sub(pattern, r"\1 \\in [\2, \3)", expr)

    # Replace AND with \land
    expr = expr.replace("AND", r"\land")

    # Wrap in $ for math mode
    return f"${expr}$"


input_str = "FEAT: [1187:2039[ AND num_modified_lines: [3:7["
output_str = rewrite_interval_expression(input_str)
print(output_str)

input_str = "FEAT_num_of_code_mentions: [9:23[ AND FEAT_num_of_hunks: [1:2["
output_str = rewrite_interval_expression(input_str)
print(output_str)

if __name__ == "__main__":
    while True:
        input_text = input("Put in the string to reformat: ")
        if input_text == "q":
            break

        latex_text = rewrite_interval_expression2(input_text)
        print(latex_text + "\n\n")

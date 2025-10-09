import codecs

from unidiff import PatchSet

from data_structures.benchmark_type_enum import BenchmarkType
from main.add_features_pipeline import AddFeaturesPipeline
from main.merge_data import BenchmarkResultsMerger

benchmark_merger = BenchmarkResultsMerger(
    BenchmarkType.VERIFIED, "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct"
)
df = benchmark_merger.get_df_with_resolved_status()
df_with_feats = AddFeaturesPipeline(input_df=df).execute()


counter_works = 0
ids_not_working = []
patches_not_working = []
patches_err = []

for index, row in df_with_feats.iterrows():
    patch_str = row["patch"]
    patch_str = codecs.decode(patch_str, "unicode_escape")
    try:
        patch_set = PatchSet.from_string(patch_str)
        counter_works += 1
    except Exception as e:
        ids_not_working.append(row["instance_id"])
        patches_not_working.append(patch_str)
        patches_err.append(e)
        continue


for i, patch in enumerate(patches_not_working):
    a_patch = patch
    a_patch_err = patches_err[i]
    a_id = ids_not_working[i]
    pass

fixed_patches_id = [
    "django__django-13809",
    "sphinx-doc__sphinx-11510",
    "django__django-12774",
    "sphinx-doc__sphinx-8548",
    "astropy__astropy-14369",
    "django__django-14349",
    "django__django-13516",
    "sphinx-doc__sphinx-7748",
    "django__django-16145",
    "django__django-12155",
    "scikit-learn__scikit-learn-25232",
    "sphinx-doc__sphinx-8593",
    "pydata__xarray-3993",
    "matplotlib__matplotlib-23412",
    "sphinx-doc__sphinx-8035",
    "sympy__sympy-13877",
    "sympy__sympy-13798",
    "astropy__astropy-14598",
    "scikit-learn__scikit-learn-14053",
    "django__django-16662",
    "scikit-learn__scikit-learn-10297",
]
fixed_patches = {}
fixed_but_not_really = 0
for i, patch in enumerate(patches_not_working):
    a_patch = patch
    a_patch_err = patches_err[i]
    a_id = ids_not_working[i]

    err_identifier = "Hunk diff line expected:"

    if not a_id in fixed_patches_id:
        print("fix me")

    if str(a_patch_err).startswith(err_identifier):
        code_part = str(a_patch_err).split(err_identifier)[1].strip()
        corrected_patch = a_patch.replace("\n" + code_part, code_part)

        try:
            PatchSet.from_string(corrected_patch)
            fixed_patches[a_id] = corrected_patch
        except Exception as e:
            fixed_but_not_really += 1

    else:
        pass

pass

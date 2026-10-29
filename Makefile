.PHONY: build check commit quality style test

# Code written for this release: held to ruff's full rule set, formatting
# included.
FORMAT_PATHS := tests \
	evaluation/input_checks.py evaluation/test_input_checks.py \
	evaluation/groundingsuite/gs_result_store.py evaluation/groundingsuite/test_gs_result_store.py \
	projects/rl/datasets/check_rl_parquet.py \
	verl/trainer/resolve_config.py generalized/verl/trainer/resolve_config.py

# The forked runtimes, the evaluation scripts, and the dataset/model code the
# documented commands reach are linted but keep their existing formatting, so
# that diffs against upstream EasyR1/veRL stay readable. projects/vlm is the
# research tree this release was carved out of and is not linted.
LINT_PATHS := $(FORMAT_PATHS) verl generalized/verl generalized/tests evaluation \
	projects/rl projects/transformers

# CPU-only suites: spatial + video runtimes and the evaluation helpers.
TEST_PATHS := tests generalized/tests evaluation

build:
	uv build

check: quality test

commit:
	pre-commit install
	pre-commit run --all-files

quality:
	ruff check $(LINT_PATHS)
	ruff format --check $(FORMAT_PATHS)

style:
	ruff check $(LINT_PATHS) --fix
	ruff format $(FORMAT_PATHS)

test:
	python3 -m pytest -q $(TEST_PATHS)

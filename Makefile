.PHONY: build check commit quality style test

# Paths held to ruff. The forked verl trees (verl/, generalized/verl/) still
# carry upstream lint debt and are excluded until they are cleaned up.
LINT_PATHS := tests \
	evaluation/input_checks.py evaluation/test_input_checks.py \
	evaluation/groundingsuite/gs_result_store.py evaluation/groundingsuite/test_gs_result_store.py \
	projects/rl/datasets/check_rl_parquet.py \
	verl/trainer/resolve_config.py generalized/verl/trainer/resolve_config.py

# CPU-only suites: spatial + video runtimes and the evaluation helpers.
TEST_PATHS := tests generalized/tests evaluation

build:
	python3 setup.py sdist bdist_wheel

check: quality test

commit:
	pre-commit install
	pre-commit run --all-files

quality:
	ruff check $(LINT_PATHS)
	ruff format --check $(LINT_PATHS)

style:
	ruff check $(LINT_PATHS) --fix
	ruff format $(LINT_PATHS)

test:
	python3 -m pytest -q $(TEST_PATHS)

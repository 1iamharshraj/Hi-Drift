.PHONY: eval_all eval_matrix test

eval_all:
	python scripts/run_eval.py

eval_matrix:
	python scripts/run_eval_matrix.py --config configs/eval/matrix_publishable.json

test:
	pytest -q

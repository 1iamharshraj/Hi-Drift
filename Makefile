.PHONY: eval_all test

eval_all:
	python scripts/run_eval.py

test:
	pytest -q


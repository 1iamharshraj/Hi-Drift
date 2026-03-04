.PHONY: eval_all eval_matrix benchmark_check publication_check paper_ready test

eval_all:
	python scripts/run_eval.py

eval_matrix:
	python scripts/run_eval_matrix.py --config configs/eval/matrix_publishable.json

benchmark_check:
	python scripts/check_benchmark_registry.py

publication_check:
	python scripts/check_publication_readiness.py

paper_ready:
	python scripts/run_eval_matrix.py --config configs/eval/matrix_publishable.json
	python scripts/export_figures.py
	python scripts/check_benchmark_registry.py
	python scripts/check_publication_readiness.py

test:
	pytest -q

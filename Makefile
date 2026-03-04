.PHONY: eval_all eval_matrix prepare_official eval_iccv iccv_check benchmark_check publication_check paper_ready test

eval_all:
	python scripts/run_eval.py

eval_matrix:
	python scripts/run_eval_matrix.py --config configs/eval/matrix_publishable.json

prepare_official:
	python scripts/prepare_official_benchmarks.py

eval_iccv:
	python scripts/prepare_official_benchmarks.py
	python scripts/run_eval_matrix.py --config configs/eval/matrix_iccv.json

iccv_check:
	python scripts/check_iccv_readiness.py

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

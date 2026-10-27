.PHONY: setup test train serve score docker-build docker-run

setup:
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e .

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

train:
	PYTHONPATH=src python -m crowe_anomaly.train

serve:
	PYTHONPATH=src python -m crowe_anomaly.service

score:
	PYTHONPATH=src python -m crowe_anomaly.score --input data/manufacturing_telemetry.csv --output artifacts/all_scored.csv

docker-build:
	docker build -t crowe-ml-anomaly-lab:local .

docker-run:
	docker run --rm -p 8000:8000 crowe-ml-anomaly-lab:local


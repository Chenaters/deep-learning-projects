# Enterprise Manufacturing Anomaly Lab

A portfolio project designed around the responsibilities in Crowe's Machine Learning Intern posting. You will frame an operational problem, compare models, train a neural-network autoencoder, package it as tested Python, deploy a scoring service, and explain the result to both engineers and business stakeholders.

The dataset is fully synthetic. The project contains no personal or employer data.

## What you will learn

| Crowe job function | Where you practice it | Evidence you can show |
|---|---|---|
| Frame business problems as data-science problems | `docs/problem_statement.md` | Explicit user, decision, cost, and success criteria |
| Navigate research, development, deployment, and support | `data.py`, `train.py`, `service.py`, `k8s/` | End-to-end lifecycle rather than a notebook-only model |
| Select and train models correctly | Chronological split, normal-only training, candidate comparison, Isolation Forest baseline | Leakage-aware experiment and model-selection table |
| Create tests and diagnostics | `tests/`, `metrics.json`, `test_scored.csv` | Precision, recall, F1, confusion counts, flags by failure type |
| Write reusable Python | `src/crowe_anomaly/` | Small modules, clear contracts, CLI entry points, type hints |
| Use Git, containers, CI/CD, and Kubernetes | `.github/workflows/`, `Dockerfile`, `k8s/` | Automated test gate, multi-stage image, rolling deployment |
| Participate in code review and agile work | `docs/backlog.md`, `docs/code_review_checklist.md` | Stories, acceptance criteria, PR checklist |
| Apply risk and compliance controls | `schema.py`, `docs/risk_register.md`, `docs/model_card.md` | PII rejection, human review, limits, rollback and monitoring plan |
| Present to technical and nontechnical audiences | `artifacts/stakeholder_summary.md`, `docs/demo_script.md` | A five-minute demo and an executive brief |

## Project architecture

```text
Synthetic sensors -> schema checks -> chronological split
                  -> normal-only scaler and autoencoder training
                  -> candidate comparison + Isolation Forest baseline
                  -> diagnostics + packaged model
                  -> batch scorer / HTTP service
                  -> Docker image -> Kubernetes Deployment
```

## Quick start on Windows

Use Python 3.12 or newer.

```powershell
cd manufacturing-anomaly-detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m crowe_anomaly.train
```

Or, after installing the dependencies, run the complete local demonstration:

```powershell
.\scripts\run_demo.ps1
```

## Quick start on Linux or macOS

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python -m crowe_anomaly.train
```

## Review the outputs

Training creates:

- `data/manufacturing_telemetry.csv`: safe synthetic input data.
- `artifacts/model.joblib`: the versioned scaler, model, threshold, and feature contract.
- `artifacts/metrics.json`: candidate comparison, baseline, test metrics, and failure-type diagnostics.
- `artifacts/test_scored.csv`: row-level test predictions for error analysis.
- `artifacts/stakeholder_summary.md`: a nontechnical decision brief.

## Score a batch

```bash
python -m crowe_anomaly.score \
  --input data/manufacturing_telemetry.csv \
  --output artifacts/all_scored.csv
```

## Run the scoring service

```bash
python -m crowe_anomaly.service --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Score one record:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"temperature_c":72.0,"vibration_mm_s":2.2,"pressure_bar":35.0,"motor_current_a":12.5,"throughput_units_h":102.0}'
```

The service rejects unexpected fields, common PII names, missing features, nonnumeric values, and nonfinite values. It does not log request bodies.

## Practice Docker and Kubernetes

```bash
docker build -t crowe-ml-anomaly-lab:local .
docker run --rm -p 8000:8000 crowe-ml-anomaly-lab:local
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Before Kubernetes deployment, replace the placeholder image in `k8s/deployment.yaml` with an image you control. Do not publish an image or create cloud resources until you intentionally choose a registry and access policy.

## Practice Git and code review

Initialize the project yourself so you understand every step:

```bash
git init
git add .
git commit -m "Build reproducible anomaly-detection baseline"
git switch -c feature/add-drift-report
```

Implement one stretch story on the feature branch, run the tests, and open a pull request using `docs/code_review_checklist.md`. Keep generated data and model artifacts out of commits; `.gitignore` already excludes them.

## Four-week learning plan

### Week 1 - Problem framing and data

1. Read `docs/problem_statement.md` and rewrite the costs using your own assumptions.
2. Generate data with three seeds and inspect feature relationships.
3. Explain why a chronological split is safer than a random split here.
4. Add one test for a malformed input.

### Week 2 - Modeling and diagnostics

1. Train the two autoencoder architectures and compare validation F1 and recall.
2. Change the threshold quantile and document the false-positive/false-negative tradeoff.
3. Compare the autoencoder with the Isolation Forest baseline.
4. Add a fourth failure pattern and report recall by anomaly type.

### Week 3 - Software engineering and deployment

1. Run the service and send valid and invalid requests.
2. Build the Docker image and inspect its layers and non-root user.
3. Open a pull request that changes one module and use the review checklist.
4. Run the CI workflow on GitHub; explain why the recall gate can block a merge.

### Week 4 - Enterprise delivery

1. Complete the model card and risk register with your experiment results.
2. Deploy locally to Kubernetes or Minikube and perform a rolling update.
3. Deliver the five-minute demo in `docs/demo_script.md`.
4. Write a retrospective: what failed, what you changed, and what you would monitor in production.

## Portfolio completion criteria

- All unit tests pass.
- The full pipeline trains reproducibly from one command.
- Test recall is at least 70% on the supplied synthetic scenario.
- You can explain why training uses normal rows only and why the threshold comes from validation normals.
- You can demonstrate a rejected PII field and a rejected malformed payload.
- The container runs as a non-root user and answers the health check.
- You can explain the Deployment's rolling-update, readiness, liveness, resource, and security settings.
- Your final brief states a decision, not just model metrics.

## Authoritative references

- [Crowe Machine Learning Intern](https://crowe.wd12.myworkdayjobs.com/External_Careers/job/Chicago-IL-USA/Machine-Learning-Intern_R-71013)
- [scikit-learn MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- [Docker build best practices](https://docs.docker.com/build/building/best-practices/)
- [GitHub Actions: building and testing Python](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)
- [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

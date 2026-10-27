# Deep Learning Projects

A collection of deep-learning projects, neural-network coursework, and related
machine-learning research software by Louis Chen. Each project has its own
setup instructions, data requirements, and validation notes.

## Deep learning and neural networks

| Project | What it explores | Validation available |
| --- | --- | --- |
| [Manufacturing anomaly detection](projects/manufacturing-anomaly-detection/) | Neural reconstruction of synthetic sensor telemetry, Isolation Forest comparison, and a scoring API | Six unit/integration tests passed |
| [RSNA knee MRI DINOv3 baseline](projects/rsna-knee-dinov3-baseline/) | Frozen vision-transformer features, study-level attention pooling, and masked multilabel loss | Three core tests passed with a dummy backbone; full training unverified |
| [IMSA deep-learning labs](projects/imsa-deep-learning-labs/) | PyTorch fundamentals, Fashion-MNIST CNN, convolutional autoencoder, and ResNet50 transfer learning | Notebook format and source syntax checked; training not rerun |
| [CUDA deep-learning benchmark](projects/cuda-deep-learning-benchmark/) | Matrix throughput and synthetic convolutional-network training on CUDA | CLI loaded; hardware-specific results intentionally excluded |
| [IMSA named entity recognition](projects/imsa-nlp-projects/) | spaCy neural NER training from CoNLL BIO annotations | Notebook format and source syntax checked; training not rerun |

## Published experiments

| Project | Included approaches | Status |
| --- | --- | --- |
| [CSIRO biomass experiments](projects/csiro-biomass-experiments/) | CNN baselines and data exploration plus small DINO regression, validation and submission helpers | Experimental coursework; full reproduction pending; known data/checkpoint-loading issues documented |

## Related machine-learning research

These R packages support prediction-model research and are included separately
from the deep-learning implementations.

| Project | Purpose | Validation available |
| --- | --- | --- |
| [pmsesampling](related-machine-learning/pmsesampling/) | Analytic OLS sample-size planning and empirical learning curves for multiple ML models | Existing R test suite passed |
| [samplesizeSimulation](related-machine-learning/samplesizeSimulation/) | Simulation-based comparisons of prediction performance across sample sizes | Existing R test suite passed |

The PMSE package credits Louis Chen and Zheyang Wu in its original metadata.
Its standalone repository is [Chenaters/pmsesampling](https://github.com/Chenaters/pmsesampling).

## Getting started

1. Open a project folder and read its README and VALIDATION.md.
2. Create a separate environment for that project; the projects use different
   frameworks and dependency versions.
3. Follow its instructions to generate synthetic data or acquire external data.
4. Run commands from that project's directory.

There is no collection-wide dependency installation. The R packages install
independently, and Python projects have their own requirements or pyproject.toml.

## Reproducibility and credits

This source collection was assembled on October 26, 2026. Tests and static checks
are recorded per project. Full GPU training, external dataset downloads, clinical
validation, container deployment, and new competition results were not performed
for this publication. Historical example metrics are labeled where retained.

IMSA exercises are labeled as coursework. Research-package author and license
declarations are preserved. See [NOTICE.md](NOTICE.md) for licensing context.
Datasets, pretrained weights, notebook outputs, local environments, credentials,
and private audit files are excluded. The CSIRO experiments are published with their limitations documented. Other
incomplete experiments remain outside this collection.

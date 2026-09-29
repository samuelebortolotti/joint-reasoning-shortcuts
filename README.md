# Shortcuts and Identifiability in Concept-based Models from a Neuro-Symbolic Lens

Code for the paper *"Shortcuts and Identifiability in Concept-based Models from a Neuro-Symbolic Lens"* by Samuele Bortolotti, Emanuele Marconato, Paolo Morettin, Andrea Passerini, and Stefano Teso, NeurIPS 2025 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fcdccd419c4dc471fa3b73ec97b53789-Abstract-Conference.html)).

Concept-based Models (CBMs) map inputs to human-interpretable concepts and then to predictions through a (possibly learned) inference layer. This repo studies **Joint Reasoning Shortcuts (JRSs)**, i.e. cases where a CBM reaches perfect training accuracy while learning concepts and/or an inference layer with unintended semantics, which hurts interpretability and out-of-distribution generalization.

It contains the CBM/NeSy-CBM implementations, the training loop, the JRS-aware evaluation metrics (concept/knowledge alignment via Hungarian matching, concept collapse), and the mitigation strategies evaluated in the paper.

## Methods and tasks

| Method | Description | Inference layer |
| --- | --- | --- |
| **DPL** | [DeepProbLog](https://arxiv.org/abs/1805.10872), probabilistic-logic reasoning | fixed (ground-truth knowledge) |
| **CBNM** | [Concept Bottleneck Model](https://arxiv.org/abs/2007.04612), no concept supervision | learned |
| **DSL** | [Deep Symbolic Learning](https://arxiv.org/abs/2208.11561), learns concepts and a truth table jointly with fuzzy logic | learned |
| **DPL\*** | DPL variant with probabilistic (not fuzzy) logic and a learned inference layer | learned |
| **bears\*** | DPL\* wrapped in a [BEARS](https://arxiv.org/abs/2402.12240)-style ensemble: one learned inference layer, several concept extractors trained against it | learned |
| **SENN** | [Self-Explaining Neural Network](https://arxiv.org/abs/1806.07538), with a reconstruction penalty | learned |

We evaluate these on four tasks: MNIST-Add, MNIST-SumParity (plus a biased/OOD variant), Clevr (restricted to 3 classes and 2-4 objects), and BDD-OIA.

## Repository structure

```
rss/
├── main.py                # entry point (training / posthoc testing / wandb sweeps)
├── datasets/               # addmnist.py, clevr.py, boia.py + BaseDataset/data-loading utils
├── models/                 # one file per method x task combination (e.g. mnistdpl.py, clevrcbm.py, boiadpl.py)
│   └── utils/               # DeepProbLog wrapper, CBM base module, MADGRAD optimizer, ...
├── backbones/               # CNN encoders/decoders used by the datasets above
├── utils/
│   ├── train.py, test.py    # training loop / posthoc evaluation
│   ├── losses.py, dpl_loss.py, contrastive.py, generative.py   # task losses + mitigation strategies
│   ├── jrs_utils.py         # Hungarian alignment, F1(C)/F1(β)/Cls(C) — the paper's JRS metrics
│   └── args.py              # CLI arguments
├── preprocessing/yolo/      # fine-tunes YOLOv11 to crop objects for Clevr (Section 8.4 of the paper)
└── data/                    # (empty, gitignored) datasets go here — see "Data" below
```

## Installation

```bash
make env && source venv/jrs/bin/activate   # optional virtualenv
make install                                # pip install -r requirements.txt
```

Tested with Python 3.9-3.12 and PyTorch >= 1.13, CPU or CUDA. `ultralytics` (YOLOv11) is only needed for the Clevr preprocessing step.

## Data

```bash
make data
```

MNIST-Add and MNIST-SumParity need nothing: `torchvision` downloads MNIST into `rss/data/raw` the first time you run something.

BDD-OIA: `make data` (or `make data-boia`) pulls the preprocessed 2048-d features used in the paper from the [rsbench Zenodo record](https://zenodo.org/records/11612556) into `rss/data/bdd2048/`. `boia.py` expects `rss/data/bdd2048/{train,val,test}_BDD_OIA.pkl`; the raw, unpreprocessed BDD-OIA videos are on the same Zenodo record if you want to redo the Faster R-CNN feature extraction yourself.

Clevr: Generate it with the `rssgen` module of [unitn-sml/rsbench-code](https://github.com/unitn-sml/rsbench-code), the same generator used for the paper. It produces `clevr/{train,val,test,ood}/` folders with `CLEVR_<split>_<id>.png` images and matching JSON metadata (see `rss/datasets/utils/clevr_creation.py` for the exact layout expected). Then fine-tune YOLOv11 on the generated boxes and crop objects — run these from `rss/`, since they import sibling packages like `datasets.*`:

```bash
cd rss
python preprocessing/yolo/yolo_finetune.py
python preprocessing/yolo/yolo_test.py
```

> Check out [`rsbench`](https://unitn-sml.github.io/rsbench/), a benchmark suite for concept quality and reasoning shortcuts published at NeurIPS 2024 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d1d11bf8299334d354949ba8738e8301-Abstract-Datasets_and_Benchmarks_Track.html)).

## Usage

Everything runs through `rss/main.py --model <model> --dataset <dataset> --task <task> [options]`. Here's how datasets/tasks map to the paper's tasks:

| Paper task | `--dataset` | `--task` |
| --- | --- | --- |
| MNIST-Add | `addmnist` | `addition` |
| MNIST-SumParity | `addmnist` | `sumparity` |
| MNIST-SumParity (biased/OOD, Table 4) | `addmnist` | `sumparityrigged` |
| Clevr | `clevr` | `clevr` |
| BDD-OIA | `boia` | `boia` |

and methods map to `--model` like this:

| Method | MNIST | Clevr | BDD-OIA |
| --- | --- | --- | --- |
| DPL | `mnistdpl` | `clevrdpl` | `boiadpl` |
| CBNM | `mnistcbm` | `clevrcbm` | `boiacbm` |
| DSL | `mnistdsl` | `clevrdsl` | — |
| DPL\* | `mnistdsldpl` | `clevrdsldpl` | — |
| bears\* | `mnistdsldpl` / `mnistdsldplrec` trained several times, see below | `clevrdsldpl` / `clevrdsldplrec` | — |
| SENN | `mnistsenn` | `clevrsenn` | — |

Most methods also have a `*rec` variant (`mnistdplrec`, `clevrcbmrec`, `mnistdsldplrec`, ...) that adds the reconstruction penalty; check `rss/models/` for exactly which combinations exist. SENN already has a reconstruction penalty built in, so it doesn't need one. bears\* isn't a model file of its own — you get it by training a `*dsldpl(rec)` model's inference layer once, freezing it, then training more concept extractors against it (see "Mitigation strategies" below and `rss/utils/jrs_utils.py`).

Train DPL on MNIST-Add for 50 epochs (from `rss/`):

```bash
cd rss
python main.py --model mnistdpl --dataset addmnist --task addition --n_epochs 50 --batch_size 64
```

Train CBNM on MNIST-SumParity:

```bash
python main.py --model mnistcbm --dataset addmnist --task sumparity --n_epochs 50 --batch_size 128
```

Train DSL on Clevr:

```bash
python main.py --model clevrdsl --dataset clevr --task clevr --n_epochs 100 --batch_size 32
```

Train DPL on BDD-OIA:

```bash
python main.py --model boiadpl --dataset boia --task boia --n_epochs 30 --batch_size 128
```

To evaluate a trained checkpoint post-hoc, add `--posthoc --checkin <path-to-ckpt>`.

### Mitigation strategies (Sections 4-5 of the paper)

| Strategy | Flags |
| --- | --- |
| Concept supervision | `--c_sup <fraction in [0,1]> --w_c <loss weight>` |
| Knowledge distillation | `--k_sup <loss weight> --perc_k <fraction of (g, y) pairs supervised>` |
| Entropy maximization | `--entropy --w_h <weight>` |
| Reconstruction penalty | use a `*rec` model, e.g. `--model mnistcbmrec`, with `--w_rec <weight>` |
| Contrastive learning (InfoNCE) | `--contrastive --w_con <weight>` |
| "Process objects individually" | always on for MNIST-Add/SumParity and Clevr, each digit/object already goes through the shared encoder separately |

Combine these however you like. For instance, a point on the `Entropy` curve of Fig. 3:

```bash
python main.py --model mnistcbm --dataset addmnist --task sumparity \
    --c_sup 0.4 --w_c 1 --entropy --w_h 1 --n_epochs 50 --batch_size 128
```

Each run writes a results CSV and a checkpoint under `rss/best_models_<dataset>/`. Aggregate those across seeds and mitigation strengths yourself to reproduce the paper's figures.

### Logging

Add `--wandb <project> --entity <your-wandb-entity>` to log to [Weights & Biases](https://wandb.ai). It's off by default, so no account is needed to train or evaluate locally.

## Evaluation metrics

`rss/utils/jrs_utils.py` implements the metrics from Section 8.6 of the paper:
- **F1(C)**, concept accuracy after aligning learned concepts to ground-truth ones with Hungarian matching on concept-wise Pearson correlation.
- **F1(β)**, knowledge accuracy: feed the learned inference layer the (permuted) ground-truth concepts and score it against the ground-truth labels.
- **Cls(C)**, concept collapse: the fraction of the concept space the model never predicts, a proxy for simplicity bias.

## Citation

```bibtex
@inproceedings{bortolotti2026shortcut,
  author = {Bortolotti, Samuele and Marconato, Emanuele and Morettin, Paolo and Passerini, Andrea and Teso, Stefano},
  booktitle = {Advances in Neural Information Processing Systems},
  doi = {10.52202/085713-5763},
  editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
  pages = {173204--173243},
  publisher = {Curran Associates, Inc.},
  title = {Shortcuts and Identifiability in Concept-based Models from a Neuro-Symbolic Lens},
  url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/fcdccd419c4dc471fa3b73ec97b53789-Paper-Conference.pdf},
  volume = {38, Main Conference},
  year = {2025}
}
```

## License

Released under the [BSD 3-Clause License](LICENSE).
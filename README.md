# FOLD

FOLD is a website fingerprinting attack on Oblivious DNS over HTTPS (ODoH). It classifies each page load in two ways, with a Transformer over the sequence of packets and a random forest over statistics of the trace that do not depend on packet order, and combines the two to identify the visited website.

This repository contains the attack, the baselines it is compared with, the scripts that run the experiments and generate the figures and tables of the paper *Characterizing Privacy Risks in Oblivious DNS over HTTPS via Website Fingerprinting*, and the ODoH traffic dataset used for the evaluation.

## Repository structure

| Path | Content |
|---|---|
| `fold/` | the attack, the baselines and the data pipeline |
| `run_experiment.py` | runs one model of one experiment |
| `run_campaign.sh` | runs all the experiments of the paper |
| `make_background.py` | builds the synthetic background traffic for the open-world experiments |
| `make_figures.py` | generates the figures and tables of the paper |
| `data/odoh_traces.csv` | the dataset: 54,155 ODoH queries from 2,834 page loads |
| `data/target_websites.csv` | the 100 monitored websites |

The dataset is also available on Zenodo, together with the raw captures: <https://doi.org/10.5281/zenodo.15115716>.

## Getting started

FOLD requires Python 3.12.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python make_background.py     # once, builds the background traffic
bash run_campaign.sh          # runs the experiments
python make_figures.py        # generates figures and tables
```

To try a single repetition first, run `REPS=1 bash run_campaign.sh`.

## Experiments

The full campaign is made of 10 repetitions of 47 jobs, 470 in total. Each job trains the models it needs, one at a time so that each training releases the GPU memory before the next one starts, then evaluates them. Every job writes to its own directory, so the jobs can run on different machines: `run_campaign.sh --list` prints all of them, and `SHARD=i/N bash run_campaign.sh` runs the i-th of N shares. The results are merged by copying the directories together.

| Directory | Jobs per repetition | Measures | In the paper |
|---|---|---|---|
| `defense_<mechanism>_p<p>` | 20 | padding, dummy injection, fragmentation and all three together, each at five probabilities, against an adversary that retrains on defended traffic | Figs. 9, 10 |
| `openworld_r<ratio>` | 5 | five proportions of traffic from websites outside the monitored set | Figs. 4, 5, 6 |
| `heads_two_r<ratio>`, `heads_n1_r<ratio>` | 10 | FOLD's two output heads against a single softmax over N+1 classes | Fig. 8 |
| `static_sweep` | 1 | a model trained on undefended traffic and evaluated under every defense | Table IV |
| `deep_fingerprinting` | 1 | Deep Fingerprinting in the closed world | Section VI-A |
| `location` | 10 | training without one client location, and a random split for reference | Table V |

Each repetition uses its own seed to shuffle the traces and the background traffic and to split them into training, validation and test sets. The seed also fixes the model initialization and the random choices of the defense, so every repetition is reproducible and all models in it see the same defended traces. Results are reported as the mean over the 10 repetitions with a 95% Student-t interval.

Each job saves the predictions of every model on every trace (`forest.npz`, `transformer.npz`, `fold.npz`, `cnn.npz`), the test labels and a `result.json` with the metrics, so any metric can be recomputed without training again. The fields of the prediction files are defined in `fold/predictions.py`.

### Running one experiment

```bash
python run_experiment.py --model forest      --experiment-id 1 --out-dir results/cw
python run_experiment.py --model transformer --experiment-id 1 --out-dir results/cw
python run_experiment.py --model fold        --experiment-id 1 --out-dir results/cw
python run_experiment.py --model evaluate    --experiment-id 1 --out-dir results/cw
```

The forest must run first, because FOLD is trained on its out-of-fold probabilities and, in the open world, the forest also selects the background traces used by the other models. `--defense-padding`, `--defense-dummy` and `--defense-fragmentation` set the per-packet probability of each countermeasure, `--open-world` with `--background-ratio` adds background traffic, and `--n1-head` replaces FOLD's two heads with a single softmax over N+1 classes. `--help` lists all the options.

## Models

| Model | Description |
|---|---|
| `fold` | the attack: a Transformer encoder combined with the random forest, with two output heads |
| `transformer` | the sequential branch alone |
| `forest` | the order-independent branch alone |
| `cnn` | the convolutional baseline from the conference version of this work |
| `df` | Deep Fingerprinting (Sirinam et al., CCS 2018), adapted to ODoH |

Deep Fingerprinting keeps its published architecture and optimizer, with two changes due to trace length: an ODoH page load has from tens to a few hundred packets, against the 5,000 Tor cells the model was designed for. Pooling uses window 2 and stride 2 instead of 8 and 4, and the classifier averages over positions instead of flattening them. `fold/models/df.py` explains both.

## Background traffic

The open-world experiments need traffic from websites outside the monitored set. This traffic is synthetic: a Gaussian copula is fitted on the statistics of the real traces and sampled to generate new websites. Real traces are used only to fit the generator and never appear in the background pool. Background traffic that is statistically close to the real one is the hardest case for the attack.

`make_background.py` builds the pool once and records its parameters in the header of the CSV file. The default parameters and seeds are the ones used for the results in the paper, so the pool is the same as long as the SDV version is the same.

## Citing

The paper describing FOLD is currently under review. Until it is published, please cite the conference version:

M. A. Salari, A. Kumar, F. Rinaudi, R. Tourani, A. Sacco and F. Esposito, *Privacy Analysis of Oblivious DNS over HTTPS: a Website Fingerprinting Study*, IEEE/IFIP DSN 2025.

## License

The code is released under the MIT License, see [LICENSE](LICENSE). The dataset in `data/` is released under CC BY 4.0, as on Zenodo.

# FOLD — website fingerprinting on Oblivious DNS over HTTPS

Code and data for *Characterizing Privacy Risks in Oblivious DNS over HTTPS via
Website Fingerprinting*.

FOLD is a website-fingerprinting attack on ODoH traffic. It reads a page load
in two ways at once — a Transformer over the sequence of packets, and a random
forest over summaries of that same trace that ignore the order — and combines
them to name the website the client visited.

This repository holds the attack, the baselines it is compared against, the
campaign that trains them all, and the figures and tables of the paper.

## What is here

| | |
|---|---|
| `fold/` | the attack, the baselines and the data pipeline |
| `run_experiment.py` | run one model of one experiment |
| `run_campaign.sh` | run the whole campaign |
| `make_background.py` | build the synthetic background pool |
| `make_figures.py` | the figures and tables of the paper |
| `data/odoh_traces.csv` | the dataset: 54 155 ODoH queries over 2834 page loads |
| `data/target_websites.csv` | the 100 monitored websites |

The dataset is also published on its own, with the raw captures, at
<https://doi.org/10.5281/zenodo.15115716>.

## Getting started

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python make_background.py            # once: the open-world background pool
bash run_campaign.sh                 # the campaign
python make_figures.py               # the figures and tables
```

A GPU is not required but the campaign is long without one. `run_campaign.sh
--list` prints the 470 jobs without running any of them, and `SHARD=i/N` runs
one machine's share, so the campaign spreads over as many machines as are
available and the results merge by copying the directories together.

To try a single experiment first:

```bash
REPS=1 bash run_campaign.sh
```

## The campaign

A job is one cell of the campaign for one repetition. Each job trains the models
it needs, one process at a time so that a training releases its GPU memory
before the next begins, and then evaluates them. Every job writes to its own
directory, keyed by the repetition, so any subset can run anywhere.

| directory | jobs per repetition | what it measures | in the paper |
|---|---|---|---|
| `defense_<mechanism>_p<p>` | 20 | padding, dummy injection, fragmentation and all three together, each at five probabilities, against an adversary that retrains on defended traffic | Figs. 9, 10 |
| `openworld_r<ratio>` | 5 | five proportions of traffic from outside the monitored set | Figs. 4, 5, 6 |
| `heads_two_r<ratio>`, `heads_n1_r<ratio>` | 10 | the two heads against a single softmax over N+1 classes | Fig. 8 |
| `static_sweep` | 1 | trained once on clean traffic, evaluated under every defense; its undefended cell is the closed-world reference | Table IV |
| `deep_fingerprinting` | 1 | Deep Fingerprinting, closed world | Section VI-A |
| `location` | 10 | leave one client location out, and a random split for reference | Table V |

Ten repetitions of 47 jobs come to 470. Each repetition reshuffles both the
monitored traces and the background pool with its own seed and carves disjoint
train/validation/test slices, so the error bars across repetitions also reflect
background sampling. Results are reported as the mean with a 95% Student-t
interval over the ten repetitions.

Every repetition seeds Python, NumPy and TensorFlow from its own identifier.
Splits, model initialisation and the stochastic defense are therefore
reproducible, and within one repetition every model sees exactly the same
defended traces.

### What a job leaves behind

```
results/openworld_r0.50/exp_001/
    forest.npz       what the forest predicted, on every split
    transformer.npz  what TR predicted
    fold.npz         what FOLD predicted
    cnn.npz          what the convolutional baseline predicted
    background.pkl   the background traces this repetition drew
    y_test.npy       the test labels
    result.json      the metrics
```

Predictions are kept per trace, not only as summary metrics, so any metric can
be recomputed afterwards without retraining. Each `.npz` holds the fields of
`fold/predictions.py` — website probabilities and, for the models that have a
head for it, the "this trace is monitored" score — and `make_figures.py` reads
exactly these files.

## Running one experiment by hand

```bash
# closed world, undefended
python run_experiment.py --model forest      --experiment-id 1 --out-dir results/cw
python run_experiment.py --model transformer --experiment-id 1 --out-dir results/cw
python run_experiment.py --model fold        --experiment-id 1 --out-dir results/cw
python run_experiment.py --model evaluate    --experiment-id 1 --out-dir results/cw
```

`forest` runs first: FOLD is trained on the forest's out-of-fold probabilities,
and in the open world the forest is also what draws the background traces every
other model reuses.

`--defense-padding`, `--defense-dummy` and `--defense-fragmentation` set the
per-packet probability of each countermeasure; `--open-world` with
`--background-ratio` mixes in background traffic; `--n1-head` replaces FOLD's
two heads with a single softmax over N+1 classes. `--help` lists the rest.

## The figures

```bash
python make_figures.py                # all of them
python make_figures.py fig4 table5    # a few
```

Output goes to `figures/`, one file per panel plus the legend as a file of its
own, at the exact width of its slot in the LaTeX source so that nothing is
rescaled on inclusion.

| | |
|---|---|
| `fig4_open_world/` | FOLD against its two branches, as the background grows |
| `fig5_open_world_cnn/` | FOLD against the convolutional baseline |
| `fig6_detection_roc/` | ROC of the monitored/background decision |
| `fig7_signal/` | which traffic properties carry the signal, and how many queries a site makes |
| `fig8_heads/` | two heads against a single softmax over N+1 classes |
| `fig9_defense_branches/` | under each countermeasure: FOLD against its branches |
| `fig10_defense_cnn/` | under each countermeasure: FOLD against the baseline |
| `table4_closed_world.tex` | closed-world performance on undefended traffic |
| `table5_location.tex` | location generalization |

Set `WF_RESULTS` to read a campaign from somewhere other than `results/`, and
`WF_FIGURES` to write the figures somewhere other than `figures/`.

## The models

| | |
|---|---|
| `fold` | the attack: the Transformer encoder fused with the forest, two heads |
| `transformer` | the sequential branch alone |
| `forest` | the order-independent branch alone |
| `cnn` | the convolutional baseline, from the conference version of this work |
| `df` | Deep Fingerprinting (Sirinam et al., CCS 2018), adapted to ODoH |

Deep Fingerprinting keeps its published architecture and optimiser. Two
deviations are deliberate and both follow from trace length — an ODoH page load
is tens to a few hundred packets against the 5000 Tor cells it was designed for:
pooling is reduced from window 8 / stride 4 to 2 / 2, and the classifier head
averages over positions instead of flattening them. `fold/models/df.py` explains
why.

## The background traffic

The open-world experiments need traffic from outside the monitored set. It is
synthetic: a Gaussian copula is fit on the statistics of the real traces and
sampled to invent new sites. Only real traces are used to fit the generator; no
real trace ever enters the pool or is labelled background. Statistically close
background is the harder case for the attack, so this is the worst case for the
adversary rather than a convenient one.

`make_background.py` writes the pool once, with its parameters recorded in the
header of the CSV it produces. The defaults are the ones behind the published
results and the seeds are pinned, so the pool is deterministic given the same
SDV version. The pool is about 110 MB and is not versioned here.

## Citing

If you use this code or the dataset, please cite:

Federico Rinaudi, Mohammad Amir Salari, Abhinav Kumar, Reza Tourani, Alessio
Sacco, Guido Marchetto and Flavio Esposito, *Characterizing Privacy Risks in
Oblivious DNS over HTTPS via Website Fingerprinting*.

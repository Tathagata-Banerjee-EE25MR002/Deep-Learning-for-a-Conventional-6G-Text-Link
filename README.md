# Exercise 1 — Deep learning for a conventional 6G text link with multi-user resource allocation

Everything needed to reproduce the figures, numbers, paper and slides of Exercise 1.

```
Exercise1/
├── code/          Python modules (one file per block, heavily commented) + run_all.py
├── notebooks/     Exercise1_Semantic_Baseline.ipynb  (same code, with explanations and outputs)
├── paper/         main.tex, main.pdf        5-page IEEE double-column conference paper
├── slides/        slides.tex / slides.pdf (simple 16:9 Beamer deck), slides_with_notes.pdf (speaker script),
│                  slides_detailed.tex / .pdf (older, more technical version)
├── results/       JSON / CSV / NPZ with every number quoted in the paper
├── figures/       paper/*.pdf (IEEE column sized) and slide/*.png
└── data/          corpus cache, trained networks (autoencoders, resource-allocation DNN), AMC look-up table
```

## Run it

```
pip install -r requirements.txt

# (A) scripts: one command
cd code
python run_all.py                    # full run: all simulations + all figures (~45 min, 2 CPU cores)
EX1_QUICK=1 python run_all.py        # smoke test (~10 min, reduced Monte-Carlo sizes)
python run_all.py --only 3 figures   # selected stages (1, 2, 3, figures)

# (B) notebook
cd notebooks
jupyter notebook Exercise1_Semantic_Baseline.ipynb
```

The notebook has three run modes (environment variable `EX1_MODE` or first code cell):

| mode | what happens | time |
|---|---|---|
| `load` (default) | uses the saved `results/` and trained networks in `data/`, regenerates all figures | ~1 min |
| `quick` | re-runs every experiment at reduced size in a separate folder `quick_run/` | ~10 min |
| `full` | re-runs everything at the size reported in the paper | ~45 min |

Every code cell `%%module <name>` of the notebook contains the verbatim source of `code/<name>.py`
(`notebooks/build_notebook.py` regenerates the notebook after a module is edited), so the notebook and the
scripts are always the same code.

The Brown corpus is downloaded automatically (3 MB) from the public `nltk_data` GitHub repository the first
time (a cached copy of the selected sentences is in `data/sentences.json`).
All randomness is seeded (`config.SEED`), so numbers are reproducible on the same library versions.
The paper PDF is built with `cd paper && pdflatex main.tex` (twice); the slides with `cd slides && pdflatex slides.tex` (twice); the speaker script with
`pdflatex -jobname=slides_with_notes "\def\withnotes{}\input{slides.tex}"` (twice).

## Code map (every file starts with a docstring that explains the design)

| file | content | task |
|---|---|---|
| `config.py` | all parameters of Table 1 + experiment sizes (single source of truth, reused by Exercise 2) | all |
| `corpus.py` | Brown corpus → clean sentences, 20 000 train / 2 000 test | 1 |
| `source_coding.py` | character-level Huffman encoder/decoder with `<EOS>` | 1 |
| `channel_coding.py` | rate-1/2 K=7 (133,171) conv. encoder, puncturing 2/3 & 3/4, vectorised **soft-decision Viterbi** | 1 |
| `modem.py` | Gray QPSK/16-QAM/64-QAM, **max-log LLR demapper**, exact BER theory (AWGN, Rayleigh) | 1 |
| `channel.py` | AWGN, Rayleigh block fading, zero-forcing equaliser | 1 |
| `link.py` | complete Tx → channel → Rx chain, batched over sentences | 1 |
| `text_metrics.py` | BLEU, word accuracy, fastText sentence similarity | 1 |
| `task1_link.py` | (a) BER validation (b) coded BER/BLER for 6 MCS (c) text quality (d) LUTs + **AMC rule** | 1 |
| `task2_autoencoder.py` | (7,4) and (2,4) autoencoders, Hamming hard/ML and BPSK baselines, Rayleigh + CSI | 2 |
| `resource_alloc.py` | system model, 4 benchmarks (round-robin, strongest+WF, **MILP**, MILP + **KKT water-filling**) | 3 |
| `dnn_ra.py` | permutation-equivariant unsupervised L2O network + hybrid variant | 3 |
| `task3_ra.py` | sweep Pmax 15…40 dBm, five metrics, AMC → correct sentences / s | 3 |
| `make_figures.py` | every figure (paper PDF and slide PNG) from the saved results | all |
| `run_all.py` | one command for everything | all |

## Modular hooks for Exercise 2 (semantic communication on the same cell)

* `config.py` defines the cell, corpus split and MCS set; `resource_alloc.generate_drops` produces the same channel drops.
* `link.TxFrames/run` is the bit-level baseline; replace the source/channel coder by a semantic encoder and reuse
  `resource_alloc` + `task3_ra.sentences_per_second` to compare in correct-sentences-per-second.
* `data/amc_lut.npz` (BLER and similarity versus SNR per MCS) is the interface between the link and the allocator.

## Honest notes

* Sentence similarity uses fastText embeddings trained on the corpus (SBERT could not be downloaded offline).
* Task 3 uses the AWGN look-up table at each RB's instantaneous SNR (the fading is constant over an RB).
* The DNN has no feasibility guarantee: about 7 % of users are unserved at 40 dBm; the KKT hybrid repairs most of it.

# Conditioning Structure, Not Entropy Alone, Predicts Memorization Dynamics

**[Project page](https://ralvarezlucend.github.io/llm-conditioning-and-memorization/)** · [Paper (PDF)](https://ralvarezlucend.github.io/llm-conditioning-and-memorization/static/paper.pdf) · [Poster](https://ralvarezlucend.github.io/llm-conditioning-and-memorization/static/poster.png)

Presented at the following NeurIPS 2026 workshops:

- [MPLR-FM](https://neurips-workshop2026.github.io/foundation_model_agentic_privacy/call_for_papers.html)
- [FLMSec](https://flmsec.github.io)
- [PriGM](https://sites.google.com/view/prigmneurips2026/call-for-papers?authuser=1)
- [ATTRIB](https://attrib-workshop.cc/#call)
- [LP4FM](https://lp4fm.github.io/#call)
- [GlobalSouthAI](https://sites.google.com/view/globalsouthai-neurips26/home)

## Abstract

Autoregressive language models do not memorize every training sequence at the same rate; yet, commonly studied predictors, such as entropy, tokenization, context length, model capacity, and repetition, do not specify how a model uses its prefix to retrieve a continuation. We study a complementary axis: the complexity of the conditioning structure that disambiguates one training sequence from the others. In controlled GPT-2-style experiments, uniformly random prefixes with the same total information content are easier to use when their information is concentrated into fewer, more distinctive tokens. We construct synthetic datasets whose suffixes are deterministic under rules of increasing complexity: a local position-aware transition, a fixed prefix cue, or a position-dependent prefix cue. The resulting memorization curves preserve this ordering across one- and four-layer models and across dataset sizes from 500 to 24,000 sequences. In one-layer models, attention patterns visibly follow the planted cue locations. However, analogous interventions on WikiText yield no detectable separation. These results support disambiguation complexity as a useful controlled axis for studying memorization and, more broadly, training dynamics.

## Setup

Requires Python 3.13+ and a GPU for the default training configuration. Run from the repository root:

```bash
uv sync
source .venv/bin/activate
```

Training logs to Weights & Biases by default. Configure your account or pass `training.report_to=none` to disable logging.

## Run an experiment

Generate the planted-cue datasets and launch their training sweep:

```bash
python generate.py one-layer-attention
python main.py --config-name=experiment/one_layer_attention_batch_1 --multirun
```

Synthetic recipes in [conf/](conf/) produce NPZ files under `datasets/` and matching training configs. Use the emitted configs to match current dataset filenames. Training configs live in [conf/experiment/](conf/experiment/); saved models and metadata go under `saved_models/`.

## Experiments and analysis

- **Conditioning hierarchy:** [generation recipe](conf/conditions.yaml) and [analysis](analysis/conditions.ipynb), with [dataset-size](conf/conditions-vary-n.yaml) and [vocabulary-size](conf/conditions-vary-v.yaml) variants.
- **Fixed prefix information:** [training config](conf/experiment/iso-prefix-info.yaml) and [analysis](analysis/prefix-iso-info-content.ipynb).
- **K-arity:** [generation recipe](conf/karity.yaml) and [analysis](analysis/karity.ipynb).
- **Natural language:** [dataset guide](natural-language/README.md) and [results](analysis/nl_summary.ipynb).
- **Attention patterns:** [analysis guide](attention-patterns/README.md).

Analysis notebooks use W&B histories; update their entity, project, and tag filters for your runs.

## Citation

If you use this work, please cite it (GitHub's "Cite this repository" button gives the same entry, from [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{alvarezlucendo2026conditioning,
  title     = {Conditioning Structure, Not Entropy Alone, Predicts Memorization Dynamics},
  author    = {Alvarez Lucendo, Rodrigo and Panwar, Madhur and Weiss, Gail and Bosselut, Antoine},
  booktitle = {NeurIPS 2026 Workshops},
  year      = {2026}
}
```

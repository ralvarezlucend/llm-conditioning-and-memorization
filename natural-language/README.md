# Natural-language experiments

Dataset-construction notebooks for *Conditioning Structure, Not Entropy Alone, Predicts Memorization Dynamics*. These experiments test whether the conditioning-structure findings from synthetic data transfer to WikiText, where many overlapping linguistic cues are available. The paper reports no detectable separation between the natural-language memorization curves; validation of the framework in natural language remains open.

## Files

- [wikidataset_translation.ipynb](wikidataset_translation.ipynb) translates WikiText-103 sequences into eight languages with `gpt-5-mini`, tokenized with `o200k_base`. It produces English, Latin-family (Spanish, Portuguese, French, Italian), and diverse-family (Russian, Chinese, Tamil, Arabic) datasets used by the [translation sweep](../conf/experiment/translation.yaml).

- [wikidataset_multilingual.ipynb](wikidataset_multilingual.ipynb) builds aligned datasets in which Latin-family and diverse-family prefixes share a **common English suffix**, used by the [multilingual sweep](../conf/experiment/multilingual.yaml).

- [paraphrase_dataset.ipynb](paraphrase_dataset.ipynb) extracts US-politics sequences from WikiText-103 and generates nine paraphrases per sequence, plus an independent control set, used by the [paraphrase sweep](../conf/experiment/paraphrase.yaml).

- [prefix_paraphrase_dataset.ipynb](prefix_paraphrase_dataset.ipynb) paraphrases only the **prefix** of each of 100 US-politics passages while preserving the suffix, with nine paraphrases per passage and an independent control set. This is the prefix-only paraphrase construction described in the paper, used by the [prefix-paraphrase sweep](../conf/experiment/prefix-paraphrase.yaml).

## Dataset paths and analysis

The notebooks use relative output paths. Save or move their `wikitext_*` output directories to the repository root, or update `tokenized_data_path` in the corresponding [dataset configs](../conf/dataset/) to their actual location.

The separate [prefix / identifier sweep](../conf/experiment/natural-language.yaml) expects `wikitext_nl`, `wikitext_translated`, `wikitext_translated_uid`, and `wikitext_translated_uid5`. These are distinct from the translation and multilingual notebook outputs and must be prepared before launching that sweep.

See the [natural-language summary](../analysis/nl_summary.ipynb) for memorization curves and [natural-language attention notebook](../attention-patterns/natural-language.ipynb) for attention diagnostics. Follow the [attention analysis guide](../attention-patterns/README.md) to adapt imports and checkpoint paths for a local checkout.

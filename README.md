# HPGPN

HPGPN is an experimental project built on top of **[un-xPass](https://github.com/ML-KULeuven/un-xPass)** for football pass receiver selection. We gratefully acknowledge and thank the un-xPass project for providing the foundation and inspiration for this work.

The main experiment is implemented in:

[`pass_selection_gnn75_contextualized_pass_history_hierarchical_attention_graphsage_euro2024_history40.run.ipynb`](pass_selection_gnn75_contextualized_pass_history_hierarchical_attention_graphsage_euro2024_history40.run.ipynb)

This notebook trains and evaluates a Hierarchical Possession-aware Graph Pointer Network (HPGPN) for pass receiver selection on Euro 2024 data.

## Project Structure

```text
HPGPN/
+-- hpgpn/
|   +-- config.py
|   +-- datasets.py
|   +-- components/
|       +-- pass_selection.py
|       +-- soccermap.py
|       +-- base.py
|       +-- utils.py
+-- stores/
|   +-- datasets/euro2024/
|   +-- model/
+-- pass_selection_gnn75_contextualized_pass_history_hierarchical_attention_graphsage_euro2024_history40.run.ipynb
```

## Environment

This project uses the same environment as **un-xPass**. Please set up and run the notebook in an un-xPass-compatible Python environment.

## Data and Model

Preprocessed Euro 2024 train/test data is stored under:

```text
stores/datasets/euro2024/
```

The saved model checkpoint is stored under:

```text
stores/model/
```

The included checkpoint is:

```text
gnn75_ctx_pass_hist_euro2024_h40_best_ep013_val0.6020_test0.6001.pt
```

The data and model artifacts are included for this experiment only. Their use may be subject to the original data provider's terms.

## Running

Open and run the notebook cells in order:

```bash
jupyter notebook pass_selection_gnn75_contextualized_pass_history_hierarchical_attention_graphsage_euro2024_history40.run.ipynb
```

If the notebook cannot find `../stores/...`, check the current working directory of the Jupyter kernel. The paths may need to be changed to `stores/...` when running from the repository root.

## Acknowledgements and License

This project builds on and adapts code from [un-xPass](https://github.com/ML-KULeuven/un-xPass), which is distributed under the Apache License 2.0.

The original un-xPass attribution and license notice should be preserved when redistributing this code.
# Explainable NLP: BERT checkpoint representation analysis

This repository preserves the original Google Colab notebooks for an experiment adapting ideas from **Feature Map Convergence Evaluation (FMCE)** to BERT based text classification. A small BERT classifier is trained on Yahoo! Answers Topics. Hidden state representations from its saved epoch checkpoints form a feature dataset, and a separate MLP predicts which checkpoint produced each representation.

## Reported result

The saved output in `notebooks/stage2_mlp_classifier.ipynb` reports **84.87% test accuracy** (0.8487, 7,500 held out feature rows) for the **25 class checkpoint epoch prediction task**. This is not the Yahoo topic classification accuracy. The 25 epoch training log records a peak topic validation accuracy of about **73.42%** (epoch 5). These are results recorded in the supplied notebooks; this repository has not rerun training.

The project description previously used “FMCS method” and “90% accuracy.” The supplied code and slides use both *FMCE* (the evaluation approach) and *FMCS* (the generated feature/label dataset), and do not substantiate a 90% test result. This README uses the measured notebook output instead.

## Repository contents

| Path | Purpose |
| --- | --- |
| `notebooks/FMCE_FMCS_BERT_Yahoo_Clean.ipynb` | Main Yahoo experiment: train mini BERT from scratch, save checkpoints, compute CQI, extract representations, and explore an MLP. |
| `notebooks/stage2_mlp_classifier.ipynb` | Separate MLP evaluation with train, validation, and test splits; source of the 84.87% result. |
| `notebooks/FMCS_Stage2_MLP.ipynb` | Earlier standalone MLP training notebook. |
| `notebooks/FMCE_NLP_IMDB_phase1.ipynb` | Earlier exploration. Despite its title, its active data loading cell uses Yahoo! Answers Topics; alternative datasets are commented out. |
| `results/yahoo_25/` | Original epoch loss and accuracy logs plus CQI marker output. |
| `docs/Explainable NLP.pptx` | Supplied presentation. |

All four notebooks are retained as received. Their saved cell outputs are included as historical records, not as proof of a fresh reproduction.

## Environment and dependencies

The original workflow uses Google Colab, Google Drive, and a GPU (the presentation names an NVIDIA T4). Use Python 3.10 or newer because the main notebook uses `int | None`. Install the Python packages in `requirements.txt` in a Colab runtime or compatible Jupyter environment:

```bash
python -m pip install -r requirements.txt
```

The notebooks also contain their original installation cells. Package versions were not locked in the source, so future library releases may require adaptation. The Yahoo dataset and `bert-base-uncased` tokenizer are downloaded by the notebooks through Hugging Face.

## Reproduction path

1. Open `notebooks/FMCE_FMCS_BERT_Yahoo_Clean.ipynb` in Colab. Mount Drive and set `CFG.base_dir`, `CFG.ckpt_root`, and `CFG.fmcs_dir` to locations you can write to. The source defaults to `/content/drive/MyDrive/ExplainableNLP/...`.
2. Run the data preparation and mini BERT training cells. The saved configuration uses 3 transformer layers, hidden width 256, 25 epochs, and Yahoo! Answers Topics. Training saves an `epoch_N` checkpoint for each epoch and writes `epoch_logs.json`.
3. Run the CQI and feature extraction cells. The recorded run did **not** meet its CQI threshold; the later extraction cell instead enumerates all `epoch_*` directories, samples 2,000 training examples per checkpoint, and creates 50,000 rows of 256 dimensional features with labels 1 through 25.
4. Before running the original feature save cell, define `feature_path` and `label_path` using `CFG.fmcs_dir` and the filenames `fmcs_features_layer-1.npy` and `fmcs_labels_layer-1.npy`. In the original notebook these assignments appear in the *following* cell, so a clean top to bottom run would otherwise fail at the save step. Run its sanity check after saving.
5. Open `notebooks/stage2_mlp_classifier.ipynb`, change `DATA_DIR` to the generated feature directory, and run the cells in order. It uses a stratified 70/15/15 row split, fits `StandardScaler` on training rows, trains an MLP for 100 epochs, and evaluates on 7,500 test rows in the recorded run.

The feature arrays and model checkpoints are stored in the [source Google Drive folder](https://drive.google.com/drive/folders/1LHlNhh1ClzTJOEk4G9g2rL45hh0QxLGE), subject to the folder's access permissions. They are excluded from Git because the checkpoint directories and some arrays are very large. The `FMCE_dataset_yahoo_25` folder contains the feature and label arrays used by the reported MLP run.

## Interpretation and limitations

- **Task distinction:** 84.87% measures checkpoint epoch classification from BERT representations. It does not measure sentiment classification, Yahoo topic prediction, or an independently validated convergence score.
- **Convergence criterion:** `results/yahoo_25/epoch_markers.json` contains only marker 25 (`E0 = EK = 25`). The main notebook's later feature builder uses all 25 epochs instead of the computed marker set.
- **Split design:** The feature builder independently samples examples at each epoch, so the same source text can appear in multiple checkpoint groups. Stage 2 splits feature *rows*, not source documents. The reported test accuracy therefore should not be interpreted as generalization to unseen texts.
- **Best checkpoint handling:** The MLP notebook stores `model.state_dict()` without copying its tensors when validation accuracy improves. The recorded final test run may therefore use later weights rather than the true best validation checkpoint.
- **Notebook state:** Several cells depend on Colab Drive paths or values defined in earlier interactive runs. The notebooks are preserved unchanged; follow the adjustment above for a fresh run.

## Reference

Zhang, L., Chen, C., He, L., & Li, K. (2024). “Feature map convergence evaluation for functional module.” *IEEE 27th International Conference on Intelligent Transportation Systems*, 864–869. This is the prior computer vision work cited in the supplied presentation, not a publication of this repository.

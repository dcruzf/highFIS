# Notebooks

One notebook for each article reproduced in [`examples/reproductions`](../reproductions).
A notebook follows one dataset of the article, one step at a time: the data, a first fit,
the setting of the article, plots of the training and of the fitted model, and a
comparison with the published result. Each runs in a few minutes and is stored with its
outputs, so it can be read without being run.

| Notebook | Family | Dataset |
|---|---|---|
| `htsk_2021.ipynb` | TSK, LogTSK, HTSK | Vowel |
| `hdfis_2023.ipynb` | HDFIS-prod, HDFIS-min | Colon |
| `adatsk_2022.ipynb` | AdaTSK | Wine |
| `fsre_adatsk_2022.ipynb` | FSRE-AdaTSK | Wine |
| `dgtsk_2023.ipynb` | DG-TSK | Wine |
| `dgaletsk_2023.ipynb` | DG-ALETSK | Colon |
| `admtsk_2025.ipynb` | ADMTSK | Colon |
| `ayatsk_2025.ipynb` | AYATSK | Wdbc |
| `adptsk_2025.ipynb` | ADPTSK | Colon |
| `mhtsk_2025.ipynb` | MHTSK | Colon |

The scripts in `examples/reproductions` run the full protocol of each article, with all
its datasets and repetitions; the notebooks are the short version.

To run a notebook, install highFIS with Jupyter and Matplotlib:

```bash
pip install highfis jupyter matplotlib
jupyter lab examples/notebooks
```

Vowel and Colon are downloaded from OpenML on the first run.

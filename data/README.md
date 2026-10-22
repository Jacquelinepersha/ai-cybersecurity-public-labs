# Data

This folder holds the **prepared training and testing partition** of UNSW-NB15, compressed with gzip so the labs run with no setup:

```
data/raw/
├── UNSW_NB15_training-set.csv.gz   (175,341 rows)
├── UNSW_NB15_testing-set.csv.gz    (82,332 rows)
└── UNSW_NB15_features.csv          (column dictionary)
```

`src/data_loader.load_unsw()` reads these files directly. If they are missing, it downloads them from this repository automatically. You can also drop the original uncompressed `.csv` files here; the loader uses them first.

Labs 18–21 download the official MITRE ATT&CK and ATLAS STIX bundles at runtime. To work offline, save `enterprise-attack.json` and `stix-atlas.json` into this folder.

## Source and terms of use

UNSW-NB15, Cyber Range Lab, UNSW Canberra:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

From the dataset page: free use for academic research purposes is granted; use for commercial purposes should be agreed with the authors; the authors (Nour Moustafa and Jill Slay) assert copyright. The dataset is **not** covered by this repository's MIT license.

## Citation

If you use UNSW-NB15, cite the authors' papers listed on the dataset page, including:

Moustafa, N. and Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* Military Communications and Information Systems Conference (MilCIS), IEEE.

# Dataset

**Online Shoppers Purchasing Intention** from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- File: `data/raw/online_shoppers_intention.csv`
- Rows: 12,330 sessions, 17 features, binary target `Revenue`
- SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`
- License: CC BY 4.0

Refresh the file with:

```bash
python scripts/fetch_data.py --force
```

The script downloads the UCI archive and refuses to write the file if the
checksum does not match.

## Citation

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of online
shoppers' purchasing intention using multilayer perceptron and LSTM recurrent
neural networks. *Neural Computing and Applications* 31, 6893-6908 (2019).
https://doi.org/10.1007/s00521-018-3523-0

The file is committed to this repository because it is small and the project is
a portfolio reference. In a production system it would live in object storage
and be versioned with DVC or a feature store.

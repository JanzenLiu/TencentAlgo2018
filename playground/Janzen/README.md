## Data Source

    .
    └── data                            # Toy Datasets
        ├── dog_breed                   # downloaded from https://www.kaggle.com/c/dog-breed-identification/data
        │   ├── Test                    # (folder) unzipped from test.zip
        |   ├── Train                   # (folder) unzipped from train.zip
        |   ├── labels.csv              # unzipped from labels.csv.zip
        |   └── sample_submission.csv   # unzipped from sample_submission.csv.zip
        └── house_pricing               # downloaded from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
            ├── data_description.txt    # directly downloaded
            ├── sample_submission.csv   # directly downloaded or unzipped from sample_submission.csv.gz                 
            ├── test.csv                # directly downloaded or unzipped from test.csv.gz
            └── train.csv               # directly downloaded or unzipped from train.csv.gz

## TODO
- [ ] Try TensorFlow
- [ ] Try PyTorch
- [ ] Try MXNet
- [x] Try FM
- [ ] Try FFM
- [ ] Try PNN (Polynominal Neural Networks)
- [ ] Try matrix decomposition (MF, SVD/SVD++, etc.)
- [ ] Try sklearn FeatureUnion and Pipeline
- [ ] Try tuning hyperparameters with hyperopt
- [ ] Try stacking
- [ ] Try ensemble with raw features involved (my idea)
- [ ] Try cython
- [ ] Try TSNE
- [x] Add simple data visualization
- [x] Add simple testing
- [ ] Implement (or just copy from somewhere) Bayesian smoothing for CTR/CVR
- [ ] Implement simple A/B test
- [ ] Implement a full tunable and testable pipeline (it's fine to be simple, but it must be complete)
- [x] Fix globals problem in info_utils

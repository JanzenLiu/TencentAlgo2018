import pandas as pd
import numpy as np
from sklearn import datasets
import sys
sys.path.append('../../../analysis')
from visualize_single_feature import SingleFeatureVisualizer

# Transform scikit-learn dataset to pandas dataframe
# Source: https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset
iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

SingleFeatureVisualizer.plot_numerical_feats(df, savepath='numerical_feats.png')

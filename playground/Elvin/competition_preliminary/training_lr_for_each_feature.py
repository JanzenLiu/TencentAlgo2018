
# coding: utf-8

# In[1]:

import lightgbm as lgb
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import pickle
import os
import gc
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression   
from sklearn import metrics


# In[2]:

data_dir = '../../../data/raw/preliminary_contest_data/'
ad_cnt_dir = '../../../data/nlp_count/preliminary_contest_data/byAdFeatureName/'
user_cnt_dir = '../../../data/nlp_count/preliminary_contest_data/byUserFeatureName/'
user_tfidf_dir = '../../../data/nlp_tfidf/preliminary_contest_data/byUserFeatureName/'


# In[3]:

def load(filename, **kw):
    return pd.read_csv(os.path.join(data_dir, filename), **kw)


# In[4]:

def load_pickle(filepath):
    obj = None
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


# In[5]:

def load_ad_cnt(feat_name):
    filename = "adFeature.[featureName='{}'].pkl".format(feat_name)
    filepath = os.path.join(ad_cnt_dir, filename)
    index, matrix = load_pickle(filepath)
    
    filename = "aid.pkl".format(feat_name)
    filepath = os.path.join(ad_cnt_dir, filename)
    uid = load_pickle(filepath)
    
    return uid, (index, matrix)


# In[6]:

def load_user_cnt(feat_name):
    filename = "userFeature.[featureName='{}'].pkl".format(feat_name)
    filepath = os.path.join(user_cnt_dir, filename)
    index, matrix = load_pickle(filepath)
    
    filename = "uid.pkl".format(feat_name)
    filepath = os.path.join(user_cnt_dir, filename)
    uid = load_pickle(filepath)
    
    return uid, (index, matrix)


# In[7]:

def load_user_tfidf(feat_name):
    filename = "userFeature.[featureName='{}'].pkl".format(feat_name)
    filepath = os.path.join(user_tfidf_dir, filename)
    index, idf, matrix = load_pickle(filepath)
    
    filename = "uid.pkl".format(feat_name)
    filepath = os.path.join(user_tfidf_dir, filename)
    uid = load_pickle(filepath)
    
    return uid, (index, idf, matrix)


# In[8]:

def get_time_str():
    return time.strftime("%H:%M:%S", time.gmtime())


# In[9]:

df_train = load("train.csv")


# In[ ]:

features = ['education','interest1','interest2','interest3','interest4','interest5',
            'kw1','kw2','kw3','appIdAction','appIdInstall']
def auc4features(feature):
    uid, (ufeat_index, uvec) = load_user_cnt(feature)
    aid, (afeat_index, avec) = load_ad_cnt('aid')
    uid_to_index = dict(zip(uid, list(range(len(uid)))))  # mapping from uids to distinct indices
    aid_to_index = dict(zip(aid, list(range(len(aid)))))  # mapping from aids to distinct indices

    a_index = df_train['aid'].map(aid_to_index).values  # list of indices for matrix joining
    u_index = df_train['uid'].map(uid_to_index).values  # list of indices for matrix joining

    X = hstack((avec[a_index,:], uvec[u_index,:])).tocsr()  # joined user and advertise matrix
    y = (df_train['label'].values + 1) / 2

    X_train, y_train = X[:6000000], y[:6000000]
    X_valid, y_valid = X[6000000:], y[6000000:]
    print("[{}] {} data prepared".format(get_time_str(), feature))

    lr = LogisticRegression(penalty = 'l2',solver = 'sag', max_iter = 100, verbose = 1)
    logisticR = lr.fit(X_train, y_train)
    print("[{}] {}: training completed".format(get_time_str(), feature))
    
    lr_pred = logisticR.predict_proba(X_valid)
    lr_auc = metrics.roc_auc_score(y_valid, lr_pred[:,1])
    print("[{}] {}: Predicting AUC is {}".format(get_time_str(), feature, lr_auc))
    
    gc.collect()
    return lr_auc


# In[ ]:

auc_scores = dict(zip(features, list(range(len(features)))))
for feature in features:
    auc_scores[feature] = auc4features(feature)
print(auc_scores)


# In[ ]:

# sort the dict based on values
import scipy.sparse as sparse
import pandas as pd
import pickle
import tqdm
import os
import gc
from scipy.sparse import hstack
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import config
from config import DataPathFormatter


# path corrector
def _correct_path(path):
    return os.path.join(BASE_DIR, path)


def user_feature_path(feat_name):
    filename = "userFeature.[featureName='{}'].data".format(feat_name)
    return os.path.join(PRELIMINARY_USER_FEATURE_DIR, filename)


# DataPathFormatter declaration
exist_in_current_repo = os.path.exists(config.DATA_DIR)
SERVER_DATA_DIR = os.path.join(config.BASE_DIR, '../../../zhangez698/TencentAlgo2018/data')
# SERVER_DATA_DIR = os.path.join(config.BASE_DIR, '../../../data')  # use this line after moving data out of elvin's dir
dpf = DataPathFormatter(None if exist_in_current_repo else SERVER_DATA_DIR)  # absolute path to /code

# paths as constants
BASE_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(BASE_PATH)
PRELIMINARY_RAW_DATA_DIR = dpf.get_path('raw', stage='prelim')
PRELIMINARY_AD_CNT_DIR = dpf.get_path('nlp_count', stage='prelim', sub_dirs=['byAdFeatureName']) 
PRELIMINARY_USER_FEATURE_DIR = dpf.get_path('split', stage='prelim', sub_dirs=['byUserFeatureName']) 
PRELIMINARY_USER_CNT_DIR = dpf.get_path('nlp_count', stage='prelim', sub_dirs=['byUserFeatureName']) 
PRELIMINARY_USER_TFIDF_DIR = dpf.get_path('nlp_tfidf', stage='prelim', sub_dirs=['byUserFeatureName'])
PRELIMINARY_USER_COOC_DIR = dpf.get_path('nlp_cooccurrence', stage='prelim', sub_dirs=['byUserFeatureName'])
PRELIMINARY_RAW_FILE_DICT = {
    "train": "train.csv",
    "test": "test1.csv",
    "ad": "adFeature.csv",
 }


# ==============
# Pickle Handler
# ==============
def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


# =======================
# Preliminary Data Loader
# =======================
def load_preliminary_raw_data(name, **kw):
    filename = PRELIMINARY_RAW_FILE_DICT[name]
    return pd.read_csv(os.path.join(PRELIMINARY_RAW_DATA_DIR, filename), **kw)


def load_preliminary_ad_cnt(feat_name):
    filename = "adFeature.[featureName='{}'].pkl".format(feat_name)
    filepath = os.path.join(PRELIMINARY_AD_CNT_DIR, filename)
    index, matrix = load_pickle(filepath)

    filename = "aid.pkl".format(feat_name)
    filepath = os.path.join(PRELIMINARY_AD_CNT_DIR, filename)
    uid = load_pickle(filepath)

    return uid, (index, matrix)


def load_preliminary_user_cnt(feat_name):
    filename = "userFeature.[featureName='{}'].pkl".format(feat_name)
    filepath = os.path.join(PRELIMINARY_USER_CNT_DIR, filename)
    index, matrix = load_pickle(filepath)

    filename = "uid.pkl".format(feat_name)
    filepath = os.path.join(PRELIMINARY_USER_CNT_DIR, filename)
    uid = load_pickle(filepath)

    return uid, (index, matrix)


def load_preliminary_user_tfidf(feat_name):
    filename = "userFeature.[featureName='{}'].pkl".format(feat_name)
    filepath = os.path.join(PRELIMINARY_USER_TFIDF_DIR, filename)
    index, idf, matrix = load_pickle(filepath)

    filename = "uid.pkl".format(feat_name)
    filepath = os.path.join(PRELIMINARY_USER_TFIDF_DIR, filename)
    uid = load_pickle(filepath)

    return uid, (index, idf, matrix)


def load_preliminary_user_feature(feat_name, **kw):
    sep = kw.pop('sep', '|')
    dtype = kw.pop('dtype', {feat_name: str})
    filepath = user_feature_path(feat_name)
    return pd.read_csv(filepath, sep=sep, dtype=dtype, **kw)


# ===================
# Playoff Data Loader
# ===================
# to be added (if we're lucky enough to enter the playoff lol)


# ======================================
# Preliminary Intermediate Result Loader
# ======================================
def load_preliminary_user_feature_coocurrence(feat_name):
    cooc_file = "userFeature.[featureName='{}'].pkl".format(feat_name)
    cooc_path = os.path.join(PRELIMINARY_USER_COOC_DIR, cooc_file)
    return load_pickle(cooc_path)


# ==========================================
# Unified Data Loader that I suggest to call
# ==========================================
def load_raw_data(name, stage="preliminary", **kw):
    if stage == "preliminary":
        return load_preliminary_raw_data(name, **kw)
    else:
        return None


def load_ad_cnt(feat_name, stage="preliminary"):
    if stage == "preliminary":
        return load_preliminary_ad_cnt(feat_name)
    else:
        return None


def load_user_cnt(feat_name, stage="preliminary"):
    if stage == "preliminary":
        return load_preliminary_user_cnt(feat_name)
    else:
        return None


def load_user_tfidf(feat_name, stage="preliminary"):
    if stage == "preliminary":
        return load_preliminary_user_tfidf(feat_name)
    else:
        return None


def load_user_feature(feat_name, stage="preliminary"):
    if stage == "preliminary":
        return load_preliminary_user_feature(feat_name)
    else:
        return None


def load_user_feature_coocurrence(feat_name, stage="preliminary"):
    if stage == "preliminary":
        return load_preliminary_user_feature_coocurrence(feat_name)
    else:
        return None


def quick_join(ad_user, user_feat_names=None, ad_feat_names=None, stage="preliminary"):
    final_mat = None
    feat_names = []

    # load and join user features
    if user_feat_names is not None and len(user_feat_names) > 0:
        for feat_name in tqdm.tqdm(user_feat_names, desc="loading user matrices"):
            uid_index, (val_to_index, cnt_feat) = load_user_cnt(feat_name, stage=stage)
            uid_to_index = dict(zip(uid_index, list(range(len(uid_index)))))
            join_index = ad_user['uid'].map(uid_to_index).values
            user_mat = cnt_feat[join_index, :]
            if final_mat is None:
                final_mat = user_mat
            else:
                final_mat = sparse.hstack((final_mat, user_mat))
                del user_mat
                gc.collect()
            feat_names += ["{}_{}".format(feat_name, val)
                           for val, index in sorted(val_to_index.items(), key=lambda x: x[1])]

            del cnt_feat
            del join_index
            del uid_to_index
            del uid_index
            del val_to_index
            gc.collect()

    # load and join ad features
    if ad_feat_names is not None and len(ad_feat_names) > 0:
        for feat_name in tqdm.tqdm(ad_feat_names, desc="loading ad matrices"):
            aid_index, (val_to_index, cnt_feat) = load_ad_cnt(feat_name, stage=stage)
            aid_to_index = dict(zip(aid_index, list(range(len(aid_index)))))
            join_index = ad_user['aid'].map(aid_to_index).values
            ad_mat = cnt_feat[join_index, :]
            if final_mat is None:
                final_mat = ad_mat
            else:
                final_mat = sparse.hstack((final_mat, ad_mat))
                del ad_mat
                gc.collect()
            feat_names += ["{}_{}".format(feat_name, val) for val in val_to_index]

            del cnt_feat
            del join_index
            del aid_to_index
            del aid_index
            del val_to_index
            gc.collect()

    assert final_mat.shape[1] == len(feat_names)
    return final_mat, feat_names


def get_set(dataframe, test, features_u_want, a_features_u_want):
    id_index_vec = []                        
    for each in features_u_want:
        id_index_vec.append(load_user_cnt(each))  # eid, (efeat_index, evec) = load_user_cnt("education")
        
    id2index = []  # mapping from uids to distinct indices
    for each in id_index_vec:
        id2index.append(dict(zip(each[0], list(range(len(each[0]))))))
        
    # list of indices for matrix joining
    index_mapper = []
    for each in id2index:
        index_mapper.append(dataframe['uid'].map(each).values)  # e_index = dataframe['uid'].map(eid_to_index).values
        
    # SAME AS ABOVE!
    aid_index_vec = []
    for each in a_features_u_want:
        aid_index_vec.append(load_ad_cnt(each))
    
    aid2index = []
    for each in aid_index_vec:
        aid2index.append(dict(zip(each[0], list(range(len(each[0]))))))
        
    aindex_mapper = []
    for each in aid2index:
        aindex_mapper.append(dataframe['aid'].map(each).values)

    temp_u = hstack([id_index_vec[i][1][1][index_mapper[i], :] for i in range((len(id_index_vec)))])  # USER NLP_COUNT
    temp_a = hstack([aid_index_vec[i][1][1][aindex_mapper[i], :] for i in range((len(aid_index_vec)))])
    X = hstack([temp_u, temp_a]).tocsr()
    # X = hstack((avec[a_index,:], evec[e_index,:], i1vec[i1_index, :], i2vec[i2_index, :], i3vec[i3_index, :],
    #            i4vec[i4_index, :], i5vec[i5_index, :], k1vec[k1_index, :], k2vec[k2_index, :], k3vec[k3_index, :],
    #            appvec[app_index, :], apivec[api_index, :])).tocsr()  # joined user and advertise matrix
    if test:
        return X
    else:
        y = (dataframe['label'].values + 1) / 2
    
    return X, y

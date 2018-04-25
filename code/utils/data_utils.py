import pandas as pd
import pickle
import os


# path corrector
def _correct_path(path):
    return os.path.join(BASE_DIR, path)


def user_feature_path(feat_name):
    filename = "userFeature.[featureName='{}'].data".format(feat_name)
    return os.path.join(PRELIMINARY_USER_FEATURE_DIR, filename)


# paths as constants
BASE_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(BASE_PATH)
PRELIMINARY_RAW_DATA_DIR = _correct_path('../../data/raw/preliminary_contest_data/')
PRELIMINARY_AD_CNT_DIR = _correct_path('../../data/nlp_count/preliminary_contest_data/byAdFeatureName/')
PRELIMINARY_USER_FEATURE_DIR = _correct_path('../../data/split/preliminary_contest_data/byUserFeatureName/')
PRELIMINARY_USER_CNT_DIR = _correct_path('../../data/nlp_count/preliminary_contest_data/byUserFeatureName/')
PRELIMINARY_USER_TFIDF_DIR = _correct_path('../../data/nlp_tfidf/preliminary_contest_data/byUserFeatureName/')
PRELIMINARY_USER_COOC_DIR = _correct_path("../../data/nlp_cooccurrence/preliminary_contest_data/byUserFeatureName/")
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

import os


def _correct_path(path):
    return os.path.abspath(os.path.join(BASE_DIR, path))


# ==========================
# Feature Names as Constants
# ==========================
USER_FEAT_NAMES = ["age", "gender", "marriageStatus", "education", "consumptionAbility", "LBS",
                   "interest1", "interest2", "interest3", "interest4", "interest5",
                   "kw1", "kw2", "kw3", "topic1", "topic2", "topic3", "appIdInstall",
                   "appIdAction", "ct", "os", "carrier", "house"]  # 23 in total

USER_SINGLE_FEAT_NAMES = ['age', 'gender', 'education', 'consumptionAbility', 'LBS',
                          'carrier', 'house']  # one user has only one value

USER_MULTI_FEAT_NAMES = ['marriageStatus', 'interest1', 'interest2', 'interest3',
                         'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                         'topic2', 'topic3', 'appIdInstall', 'appIdAction', 'ct', 'os']  # 16 in total

AD_FEAT_NAMES = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize',
                 'adCategoryId', 'productId', 'productType']

# ==================
# Paths as Constants
# ==================
BASE_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(BASE_PATH)

DATA_DIR = _correct_path('../data')
INPUT_DIR = _correct_path('../data/input')
PRELIMINARY_CONTEST_DATA_SUBDIR = '/preliminary_contest_data'

RAW_DATA_DIR = '{}/raw'.format(DATA_DIR)
SPLIT_DATA_DIR = '{}/split'.format(DATA_DIR)
COUNTER_DATA_DIR = '{}/counter'.format(DATA_DIR)
VOCAB_DATA_DIR = '{}/vocabulary'.format(DATA_DIR)
NLP_COUNT_DATA_DIR = '{}/nlp_count'.format(DATA_DIR)
NLP_TFIDF_DATA_DIR = '{}/nlp_tfidf'.format(DATA_DIR)
NLP_COOC_DATA_DIR = '{}/nlp_cooccurrence'.format(DATA_DIR)

PRELIM_RAW_DATA_DIR = '{}{}'.format(RAW_DATA_DIR, PRELIMINARY_CONTEST_DATA_SUBDIR)
PRELIM_SPLIT_DATA_DIR = '{}{}'.format(SPLIT_DATA_DIR, PRELIMINARY_CONTEST_DATA_SUBDIR)
PRELIM_COUNTER_DATA_DIR = '{}{}'.format(COUNTER_DATA_DIR, PRELIMINARY_CONTEST_DATA_SUBDIR)
PRELIM_VOCAB_DATA_DIR = '{}{}'.format(VOCAB_DATA_DIR, PRELIMINARY_CONTEST_DATA_SUBDIR)
PRELIM_NLP_COUNT_DATA_DIR = '{}{}'.format(NLP_COUNT_DATA_DIR, PRELIMINARY_CONTEST_DATA_SUBDIR)
PRELIM_NLP_TFIDF_DATA_DIR = '{}{}'.format(NLP_TFIDF_DATA_DIR, PRELIMINARY_CONTEST_DATA_SUBDIR)
PRELIM_NLP_COOC_DATA_DIR = '{}{}'.format(NLP_COOC_DATA_DIR, PRELIMINARY_CONTEST_DATA_SUBDIR)

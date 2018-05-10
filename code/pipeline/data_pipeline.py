import scipy.sparse as sparse
import os
import gc
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import data_utils as du
import config


input_folder = config.INPUT_DIR


class Data:
    def __init__(self, folder, *middle_names):
        assert os.path.exists(folder)
        self.folder = folder
        self.middle_names = middle_names
        self.suffix = "{}.pkl".format(".".join(list(middle_names)))

    def get_path(self, dataset="train"):
        file = "{}.{}".format(dataset, self.suffix)
        path = os.path.join(self.folder, file)
        return path

    def load(self, dataset="train"):
        path = self.get_path(dataset)
        return du.load_pickle(path)


class DataManager:
    def __init__(self, folder=input_folder):
        assert os.path.exists(folder)
        self.folder = folder

    def build_data(self, *middle_names):
        return Data(self.folder, *middle_names)


class CrossBinaryDataManager:
    @staticmethod
    def build_data(ad_feat_name, user_feat_name):
        folder = os.path.join(config.PRELIM_NLP_COUNT_DATA_DIR,
                              "simple_cross",
                              "byUserFeatureName",
                              "[featureName='{}']".format(user_feat_name))
        middle_names = ("[adFeatureName='{}']".format(ad_feat_name), "binary")
        return Data(folder, *middle_names)


class DataUnion:
    def __init__(self, *data_instances):
        self.data_instances = data_instances

    def load(self, dataset="train"):
        cols = []
        matrix = None
        for data in self.data_instances:
            # load data
            cols_new, matrix_new = data.load(dataset)

            # update column names
            cols += cols_new

            # update matrix
            if matrix is None:
                matrix = matrix_new
            else:
                assert matrix.shape[0] == matrix_new.shape[0]
                matrix = sparse.hstack((matrix, matrix_new))
                del matrix_new
                gc.collect()

        return cols, matrix


from sklearn import metrics
import pandas as pd
import numpy as np


def online_auc(selector, y_true, y_pred, selector_name="aid", ret_verbose=False):
    assert selector.shape[0] == y_true.shape[0]
    assert selector.shape[0] == y_pred.shape[0]
    assert np.isnan(selector).sum() == 0

    select_vals = np.unique(selector)
    aucs = np.zeros(len(select_vals))

    for i, select_val in enumerate(select_vals):
        mask = (selector == select_val)
        y_true_selected = y_true[mask]
        y_pred_selected = y_pred[mask]
        aucs[i] = metrics.roc_auc_score(y_true_selected, y_pred_selected)

    if ret_verbose:
        df = pd.DataFrame({selector_name: select_vals, "auc": aucs})
        return df
    else:
        return aucs.mean()


def build_online_auc_eval(train_selectors, valid_selectors):
    def func(y_true, y_pred):
        if y_true.shape[0] == train_selectors.shape[0]:
            return online_auc(selector=train_selectors, y_true=y_true, y_pred=y_pred, ret_verbose=False)
        elif y_true.shape[0] == valid_selectors.shape[0]:
            return online_auc(selector=valid_selectors, y_true=y_true, y_pred=y_pred, ret_verbose=False)
        else:
            raise ValueError("Mismatched length: {} (train: {}, valid: {})".format(y_true.shape[0],
                                                                                   train_selectors.shape[0],
                                                                                   valid_selectors.shape[0]))

    return func


def build_lightgbm_online_auc_eval(train_selectors, valid_selectors):
    def func(y_pred, lgbm_data):
        y_true = lgbm_data.get_label()
        if y_true.shape[0] == train_selectors.shape[0]:
            auc = online_auc(selector=train_selectors, y_true=y_true, y_pred=y_pred, ret_verbose=False)
        elif y_true.shape[0] == valid_selectors.shape[0]:
            auc = online_auc(selector=valid_selectors, y_true=y_true, y_pred=y_pred, ret_verbose=False)
        else:
            raise ValueError("Mismatched length: {} (train: {}, valid: {})".format(y_true.shape[0],
                                                                                   train_selectors.shape[0],
                                                                                   valid_selectors.shape[0]))
        return "online_auc", auc, True

    return func

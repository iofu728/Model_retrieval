# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-24 21:18:48
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-24 21:25:31
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import threading
import re

from datetime import datetime
from numba import jit
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

from utils.utils import begin_time, end_time, dump_bigger, load_bigger

data_path = 'lightgbm/data/'
model_path = 'lightgbm/model/'
pickle_path = 'lightgbm/pickle/'
prediction_path = 'lightgbm/prediction/'


class Lightgbm(object):
    """
    data minie for classify
    """

    def __init__(self, do_pre=False):
        self.version = datetime.now().strftime("%m%d%H%M")
        self.seed = 333
        self.EARLY_STOP = 300
        self.OPT_ROUNDS = 2444
        self.MAX_ROUNDS = 300000
        self.evaluate_num = 0
        self.basic_auc = 0

        self.params = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.01,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 255,
            'subsample': 0.85,
            'subsample_freq': 10,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'num_leaves': 63,
            'seed': self.seed,
            'nthread': 20,
            'metric': "None",
            "verbose": -1
        }
        self.load_data_constant()

    def pre_data_list(self, do_pre):
        version = begin_time()
        if do_pre == True:
            self.load_all(0)
            self.load_all(1)
        elif do_pre == 2:
            self.load_all_pickle(0)
            self.load_all_pickle(1)
        else:
            self.load_basic(1)
        end_time(version)

    @jit
    def fast_f1(self, result, predict, true_value):
        """
        F1
        """
        true_num = 0
        recall_num = 0
        precision_num = 0
        for index, values in enumerate(result):
            if values == true_value:
                recall_num += 1
                if values == predict[index]:
                    true_num += 1
            if predict[index] == true_value:
                precision_num += 1

        R = true_num / recall_num if recall_num else 0
        P = true_num / precision_num if precision_num else 0
        return (2 * P * R) / (P + R) if (P + R) else 0

    @jit
    def fast_auc(self, y_true, y_prob):
        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_prob)]
        nfalse = 0
        auc = 0
        n = len(y_true)
        for i in range(n):
            y_i = y_true[i]
            nfalse += (1 - y_i)
            auc += y_i * nfalse
        auc /= (nfalse * (n - nfalse))
        return auc

    def eval_auc(self, preds, dtrain):
        labels = dtrain.get_label()
        return 'auc', self.fast_auc(labels, preds), True

    def load_data_constant(self):
        """
        load data constant coloums
        """
        df = pd.read_csv(data_path + 'train.csv')
        df_nunique = df.nunique()
        df_columns = [
            index for index in df_nunique.index if df_nunique[index] != 1]
        df = pd.DataFrame(df, columns=df_columns)

        df_02 = df.quantile(0.2)
        df_08 = df.quantile(0.8)
        with open(model_path + 'columns.csv', 'r') as f:
            str_f = f.readline()
            if str_f[-1] == '\n':
                str_f = str_f[:-1]
            good_columns = str_f.split(',')

        self.df_columns = df_columns
        self.basic_columns = basic_columns[:-1][1:]
        self.good_columns = good_columns

        self.wait_columns = good_columns[1:]

    def pre_data_v1(self, types):
        """
        : result to .pickle
        pre data function
        """
        file_type = 'train' if types == 1 else 'test'
        file_address = pickle_path + file_type + '.pickle'
        df = pd.read_pickle(file_address)

        df.to_pickle(pickle_path + file_type + '.pickle')

    def pre_data(self, pre, slices):
        """
        prepare data one by one from wait_columns
        """
        if slices is None:
            wait_columns = self.good_columns
            pre = pd.DataFrame(pre, columns=wait_columns)
            return pre
        else:
            wait_columns = self.good_columns
            if slices != -1:
                wait_columns = [*wait_columns, self.wait_columns[slices]]
            wait = pd.DataFrame(pre, columns=wait_columns)
            return wait

    def load_data(self, model=True, slices=None):
        """
        load data for appoint model
        @param: model True-train False-predict
        """
        print('Load data...')

        if model:
            pre = pd.read_pickle(pickle_path + 'train.pickle')
            target = pre['TARGET'].values
            pre = pre.drop(['TARGET'], axis=1)
            data = self.pre_data(pre, slices)
            data = pre
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.25)
            print('data split end')
        else:
            pre = pd.read_pickle(pickle_path + 'train.pickle')
            target = pre['TARGET'].values
            pre = pre.drop(['TARGET'], axis=1)
            X_train = self.pre_data(pre, slices)
            # X_train = pre
            y_train = target

            pre = pd.read_pickle(pickle_path + 'test.pickle')
            target = pre['TARGET'].values
            pre = pre.drop(['TARGET'], axis=1)
            X_test = self.pre_data(pre, slices)
            # X_test = pre
            y_test = target
            print('data split end')

        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train

    def train_model(self):
        """
        train model by lightgbm
        """

        print('Start training...')

        categorical = []

        dtrain = lgb.Dataset(self.X_train,
                             label=self.y_train,
                             feature_name=list(self.X_train.columns),
                             categorical_feature=categorical)

        model = lgb.train(self.params,
                          dtrain,
                          num_boost_round=self.OPT_ROUNDS,
                          valid_sets=[dtrain],
                          valid_names=['train'],
                          verbose_eval=100,
                          feval=self.eval_auc)

        importances = pd.DataFrame({'features': model.feature_name(),
                                    'importances': model.feature_importance()})

        importances.sort_values('importances', ascending=False, inplace=True)

        model.save_model(model_path + '{}.model'.format(self.version))
        importances.to_csv(
            model_path + '{}_importances.csv'.format(self.version), index=False)

        self.gbm = model
        self.dtrain = dtrain

    def evaulate_model(self, model=True):
        """
        evaulate model by lightgbm
        """
        print('Start predicting...')

        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration)

        # print(self.auc_max_index)
        # predict = [int(index > self.auc_max_index) for index in y_pred]
        print(self.fast_auc(self.y_test, y_pred))
        with open(model_path + 'result', 'a') as f:
            f.write(str(self.fast_auc(self.y_test, y_pred)) + '\n')
        if not model:
            result = pd.DataFrame({'sample_file_name': self.X_test.ID, 'label': y_pred}, columns=[
                'sample_file_name', 'label'])
            result.to_csv(prediction_path +
                          '{}.csv'.format(self.version), index=False)

    def optimize_model(self, model, index=None):
        """
        optimize model by lightgbm
        """
        # print('Feature importances:', list(self.gbm.feature_importance()))
        print(self.X_train.iloc[0, ], self.X_train.columns, len(
            self.X_train.columns), self.y_train[0])
        dtrain = lgb.Dataset(self.X_train,
                             label=self.y_train,
                             feature_name=list(self.X_train.columns),
                             categorical_feature=[])

        eval_hist = lgb.cv(self.params,
                           dtrain,
                           nfold=8,
                           num_boost_round=self.MAX_ROUNDS,
                           early_stopping_rounds=self.EARLY_STOP,
                           verbose_eval=50,
                           seed=self.seed,
                           shuffle=True,
                           feval=self.eval_auc,
                           metrics="None"
                           )
        result = [self.version]
        result.append('best n_estimators:' + str(len(eval_hist['auc-mean'])))
        result.append('best cv score:' + str(eval_hist['auc-mean'][-1]) + '\n')
        with open(model_path + 'result', 'a') as f:
            f.write('\n'.join([str(index) for index in result]))
        print('best n_estimators:', len(eval_hist['auc-mean']))
        print('best cv score:', eval_hist['auc-mean'][-1])
        self.OPT_ROUNDS = len(eval_hist['auc-mean'])
        if (eval_hist['auc-mean'][-1] > self.basic_auc):
            self.basic_auc = eval_hist['auc-mean'][-1]
            if not index is None and index != -1:
                self.good_columns.append(self.wait_columns[index])
        with open(model_path + 'columns.csv', 'w') as f:
            f.write(','.join([str(index) for index in self.good_columns]))


if __name__ == '__main__':
    version = begin_time()

    model = False
    single = True
    im = SA()
    # im.pre_data_v1(1)
    # im.pre_data_v1(0)
    # single = True
    if single:
        im.load_data(model)
        im.optimize_model(model)
        im.train_model()
        im.evaulate_model(model)

    else:
        for index in range(-1, len(im.wait_columns)):  # filter good feature
            im.load_data(model, index)
            im.optimize_model(model, index)
            im.train_model()
            im.evaulate_model(not model)

    end_time(version)

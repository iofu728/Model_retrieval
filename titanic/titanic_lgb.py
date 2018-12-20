# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-12-20 11:27:21
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-12-20 19:29:04
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings

from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class titanic_lgb(object):
    """
    train for titanic by lightbgm
    """

    def data_pre(self, wait):
        """
        pre data
        """
        wait.Sex = wait.Sex.map(lambda sex: int(sex == 'male'))
        wait.Ticket = wait.Ticket.map(
            lambda tickle: 888 if tickle == 'LINE' else int(tickle.split(' ')[-1]))
        wait.Fare = wait.Fare.fillna(wait.Fare.mean())
        wait.Age = wait.Age.fillna(wait.Age.mean())
        wait.Embarked = wait.Embarked.fillna('S').map(
            lambda embarked: ord(embarked) - ord('A'))
        test = wait.Cabin.fillna('N').map(
            lambda cabin: cabin[0]).replace('T', 'N')
        test = test.map(lambda cabin: ord(cabin) - ord('A'))
        seat = pd.get_dummies(test, prefix="Cabin")

        wait.Cabin = wait.Cabin.fillna('N').map(
            lambda cabin: int(cabin[1:]) if cabin[1:].isnumeric() else 0)

        family = pd.DataFrame()
        family['family'] = wait.SibSp + wait.Parch + 1

        titleDf = pd.DataFrame()
        titleDf['Title'] = wait.Name.map(lambda name: name
                                         .split(',')[1].split('.')[0].strip())
        titleDict = {
            'Capt': 'officer',
            'Col': 'officer',
            'Major': 'officer',
            'Dr': 'officer',
            'Rev': 'officer',
            'Jonkheer': 'Royalty',
            'Don': 'Royalty',
            'Sir': 'Royalty',
            'the Countess': 'Royalty',
            'Dona': 'Royalty',
            'Lady': 'Royalty',
            'Mlle': 'Miss',
            'Miss': 'Miss',
            'Mr': 'Mr',
            'Mme': 'Mrs',
            'Ms': 'Mrs',
            'Mrs': 'Mrs',
            'Master': 'Master'
        }
        titleDf['Title'] = titleDf['Title'].map(titleDict)
        titleDf = pd.get_dummies(titleDf['Title'])

        del wait['Name']
        del wait['PassengerId']
        wait = pd.concat([family, wait, seat, test, titleDf], axis=1)
        return wait.values

    def load_data(self, model=True):
        """
        load data for appoint model
        @param: model True-train False-predict
        """

        print('Load data...')

        if model:
            pre = pd.read_csv('data/train.csv')

            target = pre.Survived.values
            del pre['Survived']
            data = self.data_pre(pre)

            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2)
        else:
            pre = pd.read_csv('data/train.csv')

            y_train = pre.Survived.values
            del pre['Survived']
            X_train = self.data_pre(pre)
            y_test = pd.read_csv('data/gender_submission.csv').Survived.values
            X_test = self.data_pre(pd.read_csv('data/test.csv'))
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train

    def train_model(self):
        """
        train model by lightgbm
        """

        print('Start training...')
        gbm = lgb.LGBMRegressor(objective='regression',
                                num_leaves=31, learning_rate=0.095, n_estimators=26)
        gbm.fit(self.X_train, self.y_train, eval_set=[
            (self.X_test, self.y_test)], eval_metric='l1', early_stopping_rounds=5)
        self.gbm = gbm

    def evaulate_model(self, model=True, segmentation=0.66, begin_id=892):
        """
        evaulate model by lightgbm
        """
        print('Start predicting...')
        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration_)
        if model:
            min_result = 1
            min_index = 0
            total_num = len(y_pred)
            block_num = 100
            for index in range(0, block_num):
                temp_result = sum(
                    np.abs((y_pred > (index / block_num)) - self.y_test)) / total_num
                print(index, temp_result)
                if temp_result < min_result:
                    min_result = temp_result
                    min_index = index
            print('min: ' + str(min_index) + ' ' + str(min_result))
        else:
            y_pred = pd.DataFrame(y_pred > segmentation)[
                0].map(lambda y: int(y))
            PassengerId = pd.DataFrame()
            PassengerId = y_pred.index.map(lambda temp_id: temp_id + begin_id)
            wait = pd.concat([pd.DataFrame(PassengerId), y_pred], axis=1)
            wait.to_csv('result.csv', index=False)

    def optimize_model(self):
        """
        optimize model by lightgbm
        """
        print('Feature importances:', list(self.gbm.feature_importances_))

        estimator = lgb.LGBMRegressor(num_leaves=31)

        param_grid = {
            'learning_rate': [0.08, 0.085, 0.09, 0.095, 0.1],
            'n_estimators': [25, 26, 27, 28, 29, 30]
        }

        gbm = GridSearchCV(estimator, param_grid)

        gbm.fit(self.X_train, self.y_train)

        print('Best parameters found by grid search are:', gbm.best_params_)


if __name__ == '__main__':
    model = False
    titanic = titanic_lgb()
    titanic.load_data(model)
    titanic.train_model()
    titanic.evaulate_model(model)
    titanic.optimize_model()

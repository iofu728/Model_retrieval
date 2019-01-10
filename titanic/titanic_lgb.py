# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-12-20 11:27:21
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-12-26 21:24:36
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings

from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import ensemble
from sklearn import model_selection

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


def drop_col_not_req(df, cols):
    df.drop(cols, axis=1, inplace=True)


class titanic_lgb(object):
    """
    train for titanic by lightbgm
    """

    def fill_missing_age(self, missing_age_train, missing_age_test):
        missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
        missing_age_Y_train = missing_age_train['Age']
        missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

        gbm_reg = ensemble.GradientBoostingRegressor(random_state=42)
        gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [
            3], 'learning_rate': [0.01], 'max_features': [3]}
        gbm_reg_grid = model_selection.GridSearchCV(
            gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
        gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
        print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
        print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
        print('GB Train Error for "Age" Feature Regressor:' +
              str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
        missing_age_test['Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
        print(missing_age_test['Age_GB'][:4])

        lrf_reg = LinearRegression()
        lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
        lrf_reg_grid = model_selection.GridSearchCV(
            lrf_reg, lrf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
        lrf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
        print('Age feature Best LR Params:' + str(lrf_reg_grid.best_params_))
        print('Age feature Best LR Score:' + str(lrf_reg_grid.best_score_))
        print('LR Train Error for "Age" Feature Regressor' +
              str(lrf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
        missing_age_test['Age_LRF'] = lrf_reg_grid.predict(missing_age_X_test)
        print(missing_age_test['Age_LRF'][:4])

        print('shape1', missing_age_test['Age'].shape, missing_age_test[[
              'Age_GB', 'Age_LRF']].mode(axis=1).shape)
        # missing_age_test['Age'] = missing_age_test[['Age_GB','Age_LRF']].mode(axis=1)
        missing_age_test['Age'] = np.mean(
            [missing_age_test['Age_GB'], missing_age_test['Age_LRF']])
        print(missing_age_test['Age'][:4])
        drop_col_not_req(missing_age_test, ['Age_GB', 'Age_LRF'])

        return missing_age_test

    def data_pre(self, wait):
        """
        pre data
        """
        wait.Sex = wait.Sex.map(lambda sex: int(sex == 'male'))
        sex_dummies_df = pd.get_dummies(
            wait['Sex'], prefix=wait[['Sex']].columns[0])

        Group_Ticket = wait.Fare.groupby(by=wait.Ticket).transform('count')
        wait.Fare = wait.Fare / Group_Ticket
        commonTicket = pd.DataFrame()
        commonTicketMap = wait.groupby(['Ticket']).size()
        commonTicket['commonTicket'] = wait.Ticket.map(
            lambda ticket: commonTicketMap[ticket] != 1)

        wait.Ticket = wait.Ticket.map(
            lambda tickle: 888 if tickle == 'LINE' else int(tickle.split(' ')[-1]))

        wait.Fare = wait[['Fare']].fillna(
            wait.groupby('Pclass').transform('mean'))
        wait['Fare_Category'] = wait.Fare.map(lambda fare: (
            ((0 if fare <= 4 else 1) if fare <= 10 else 2) if fare <= 30 else 3) if fare <= 45 else 4)

        fare_cat_dummies_df = pd.get_dummies(
            wait['Fare_Category'], prefix=wait[['Fare_Category']].columns[0])

        wait['Family_Size'] = wait.SibSp + wait.Parch + 1
        wait['Family_Size_Category'] = wait.Family_Size.map(lambda size: (
            'Single' if size <= 1 else'Small_Family') if size <= 3 else 'Large_Family')
        le_family = LabelEncoder()
        le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
        wait['Family_Size_Category'] = le_family.transform(
            wait.Family_Size_Category)
        fam_size_cat_dummies_df = pd.get_dummies(
            wait.Family_Size_Category, prefix=wait[['Family_Size_Category']].columns[0])

        wait = pd.concat([wait, fam_size_cat_dummies_df], axis=1)

        wait['Title'] = wait.Name.map(lambda name: name
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
        wait['Title'] = wait['Title'].map(titleDict)
        titleDf = pd.get_dummies(wait['Title'])

        missing_age_df = pd.DataFrame(wait[['Age', 'Parch', 'Sex', 'SibSp', 'Family_Size', 'Family_Size_Category',
                                            'Title', 'Fare', 'Fare_Category', 'Pclass', 'Embarked']])
        missing_age_df = pd.get_dummies(missing_age_df, columns=[
                                        'Title', 'Family_Size_Category', 'Fare_Category', 'Sex', 'Pclass', 'Embarked'])
        missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
        missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
        wait.loc[(wait.Age.isnull()), 'Age'] = self.fill_missing_age(
            missing_age_train, missing_age_test)

        wait.Embarked = wait.Embarked.fillna('S').map(
            lambda embarked: ord(embarked) - ord('A'))
        test = wait.Cabin.fillna('N').map(
            lambda cabin: cabin[0]).replace('T', 'N')
        test = test.map(lambda cabin: ord(cabin) - ord('A'))
        seat = pd.get_dummies(test, prefix="Cabin")

        wait['Cabin_Null'] = wait.Cabin.isnull()
        wait['Cabin_Letter'] = wait.Cabin.fillna('N').map(
            lambda cabin: cabin[0] if cabin[1:].isnumeric() else 0)
        wait = pd.get_dummies(wait, columns=['Cabin_Letter'])
        wait.Cabin = wait.Cabin.fillna('N').map(
            lambda cabin: int(cabin[1:]) if cabin[1:].isnumeric() else 0)

        titleDf = pd.DataFrame()

        del wait['Name']
        del wait['PassengerId']
        del wait['Title']
        wait = pd.concat(
            [wait, seat, test, titleDf, commonTicket, sex_dummies_df, fare_cat_dummies_df], axis=1)
        return wait.values

    def load_data(self, model=True):
        """
        load data for appoint model
        @param: model True-train False-predict
        """

        print('Load data...')

        if model:
            pre = pd.read_csv('titanic/data/train.csv')

            target = pre.Survived.values
            del pre['Survived']
            data = self.data_pre(pre)

            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.18)
            # X_train = data
            # y_train = target
        else:
            pre = pd.read_csv('titanic/data/train.csv')

            y_train = pre.Survived.values
            del pre['Survived']
            X_train = self.data_pre(pre)
            y_test = pd.read_csv(
                'titanic/data/gender_submission.csv').Survived.values
            X_test = self.data_pre(pd.read_csv('titanic/data/test.csv'))
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
                                num_leaves=31, learning_rate=0.095, n_estimators=29)
        gbm.fit(self.X_train, self.y_train, eval_set=[
            (self.X_test, self.y_test)], eval_metric='l1', early_stopping_rounds=5)
        self.gbm = gbm

    def evaulate_model(self, model=True, segmentation=0.533, begin_id=892):
        """
        evaulate model by lightgbm
        """
        print('Start predicting...')
        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration_)
        return y_pred
        if model:
            min_result = 1
            min_index = 0
            total_num = len(y_pred)
            block_num = 1000
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
            wait.to_csv('titanic/result.csv', index=False)


y_pred = pd.DataFrame(y_pred)[0].map(
    lambda y: 0 if y < 0.51 or (y > 0.516 and y < 0.517) else 1)
PassengerId = pd.DataFrame()
PassengerId = y_pred.index.map(lambda temp_id: temp_id + begin_id)
wait = pd.concat([pd.DataFrame(PassengerId), y_pred], axis=1)
wait.to_csv('titanic/result.csv', index=False)

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

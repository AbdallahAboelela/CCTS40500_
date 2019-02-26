### ML MIDTERM ###
### ABDALLAH ABOELELA (AKABOELELA) ###

import pandas as pd 
import numpy as np
import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

# TO DO:
# 1. Add fraction of time sick <6 months, <2 years, overall
# 2. Combine files into one huge dataframe with categories for each value?
# 3. Consider only first x weeks?
# 4. Merge overall

def read_file(fname):
    with open(fname) as f:
        content = f.readlines()
    
    content = [x.strip().split(' ') for x in content]

    max_cols = 0
    for i, line in enumerate(content):
        max_cols = max(max_cols, len(line))

    name = fname[5:]

    columns = ['p_id', 'c_id'] + list(range(max_cols - 2))

    df = pd.read_csv(fname, names = columns, engine = 'python', 
        delim_whitespace = True)

    df['sex'] = df.p_id.apply(lambda x: x[0])
    df['diagnosis'] = df.p_id.apply(lambda x: x[1:4])
    df['state'] = df.p_id.apply(lambda x: x[4:6])
    df['county'] = df.p_id.apply(lambda x: x[6:9])

    for col in df.columns[2:]:
        df[col] = pd.Categorical(df[col]).codes

    return df

def classify(csv_file, directory):
    rows = [['Disease', 'NB', 'DT', 'RF', 'SVC', 'LR']]

    for fname in os.listdir('data'):
        if '.' in fname or '_' in fname or fname == 'Hepatic':
            pass

        else:
            print(fname)
            df = read_file('data/' + fname)

            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 2:].drop('diagnosis', 
                axis = 1), df.diagnosis, random_state = 42)

            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            nb_auc = roc_auc_score(y_test, y_pred)
            print('NB: ', nb_auc)

            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            dt_auc = roc_auc_score(y_test, y_pred)
            print('DTC: ', dt_auc)


            rf = RandomForestClassifier(n_estimators = 100)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_auc = roc_auc_score(y_test, y_pred)
            print('RFC: ', rf_auc)

            svc = SVC(gamma = 'auto')
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            svc_auc = roc_auc_score(y_test, y_pred)
            print('SVC: ', svc_auc)

            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            reg_auc = roc_auc_score(y_test, y_pred)
            print('LinReg: ', reg_auc)
            print()

            rows.append([fname, nb_auc, dt_auc, rf_auc, svc_auc, reg_auc])

    with open(csv_file, 'wb') as f:
        writer = csv.writer(f, delimiter=',')

        for row in rows:
            writer.writerow(row)

'''
YOU'VE BEEN DOING CALCULATIONS ALL WRONG:
In [118]: sum((ys.diagnosis == 1) & (ys.diag_pred == 1))/sum(ys.diagnosis == 1)
In [119]: sum((ys.diagnosis == 1) & (ys.diag_pred == 1))/sum(ys.diag_pred == 1)
'''


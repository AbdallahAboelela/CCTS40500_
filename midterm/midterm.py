### ML MIDTERM ###
### ABDALLAH ABOELELA (AKABOELELA) ###

import pandas as pd 
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

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

def merge_dfs(directory):
    for fname in os.listdir('data'):
        if '.' in fname or '_' in fname:
            pass

        else:
    pass

def apply_naive_bayes(directory):
    for fname in os.listdir('data'):
        if '.' in fname or '_' in fname:
            pass

        else:
            print(fname)
            df = read_file('data/' + fname)

            X_train, X_test, y_train, y_test = train_test_split(df.ix[:, 2:].drop('diagnosis', 
                axis = 1), df.diagnosis, random_state = 42)

            clf = GaussianNB()
            clf.fit(X_train, y_train)
            
            print(clf.score(X_test, y_test))
            print()












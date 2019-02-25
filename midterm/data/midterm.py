### ML MIDTERM ###
### ABDALLAH ABOELELA (AKABOELELA) ###

import pandas as pd 
import numpy as np

def read_file(fname):
    with open(fname) as f:
        content = f.readlines()
    
    content = [x.strip().split(' ') for x in content]

    max_cols = 0
    for i, line in enumerate(content):
        max_cols = max(max_cols, len(line))

    columns = ['p_id', 'c_id'] + [fname + str(x) for x in range(max_cols - 2)]

    df = pd.read_csv(fname, names = columns, engine = 'python', 
        delim_whitespace = True)

    return df
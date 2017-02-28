# Author Taher Ahmadi

import pandas as pd

def read(path, columns):
    data = pd.read_csv(path, header=None)
    labels = data.get(columns)
    data = data.drop(columns, axis=1)
    return data, labels
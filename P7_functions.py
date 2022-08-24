import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------- Preprocessing functions --------------------------------------------

def select_one_mode_value(x):
    """ After an aggregation of mode values in a dataframe for a given variable, this function selects only one mode value if several are returned during aggregation"""

    if isinstance(x, str): # If we have only one value, the type is a str. We keep this value
        return x

    elif isinstance(x, np.ndarray):
        if x.size == 0: # If the value is a NaN we have an empty array
            return np.nan
        else: # If we have several value, it's stored in a nparray, we take only the first value of this array
            return x[0]
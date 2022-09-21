from flask import Flask
from sklearn.model_selection import train_test_split
from config import *

import os
import pandas as pd
import flask_sqlalchemy
import sqlite3


# Specify path of our database file
basedir = os.path.abspath(os.path.dirname('app.py'))
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'train_set.db')

# Create engine object
engine = flask_sqlalchemy.sqlalchemy.engine.create_engine(SQLALCHEMY_DATABASE_URI)

# Following instructions performed only once

# Load datas we will use for our database
# x = pd.read_csv('../Clean_datas/real_x.csv')
# x.index = x['SK_ID_CURR']
# x.drop('SK_ID_CURR', axis=1, inplace=True)

# # If we want to reduce RAM and memory consumption
# x, _ = train_test_split(x, train_size=SUBSET_SIZE, random_state=RANDOM_STATE)

# # Link database file with these datas
# x.to_sql('train_set', con=engine, if_exists='fail', index=True, index_label='SK_ID_CURR')
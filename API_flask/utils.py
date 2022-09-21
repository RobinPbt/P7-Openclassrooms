from models import engine
import pandas as pd

def find_customer_features(id_customer):
    customer_features = engine.execute("SELECT * FROM train_set WHERE SK_ID_CURR = {}".format(id_customer)).fetchall()
    customer_features = pd.Series(customer_features[0])
    customer_features.drop('SK_ID_CURR', inplace=True)
    
    return customer_features

def find_feature_distribution(feature_name):
    feature_distribution = engine.execute("SELECT AMT_INCOME_TOTAL FROM train_set").fetchall()
    feature_distribution = [i[0] for i in feature_distribution]
    
    return feature_distribution
    
def get_full_train_set():
    full_train_set = engine.execute("SELECT * FROM train_set").fetchall()
    full_train_set = pd.DataFrame(full_train_set)
    full_train_set.index = full_train_set['SK_ID_CURR']
    full_train_set.drop('SK_ID_CURR', axis=1, inplace=True)
    
    return full_train_set
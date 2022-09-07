# Classic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import timeit

# Preprocessing
from sklearn.model_selection import KFold, StratifiedKFold

# Sklearn models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Evaluation
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier

# ----------------------------------------------- Preprocessing functions --------------------------------------------

def select_one_mode_value(x):
    """ 
    After an aggregation of mode values in a dataframe for a given variable, 
    this function selects only one mode value if several are returned during aggregation.
    Use this function with apply on a pd.Series with modes values for each row (ex : df[var].apply(lambda x: select_one_mode_value(x)))
    
    Parameters
    ----------
    - x : one-row result from a pandas mode aggregation (list, str)
    """

    if isinstance(x, str): # If we have only one value, the type is a str. We keep this value
        return x

    elif isinstance(x, np.ndarray):
        if x.size == 0: # If the value is a NaN we have an empty array
            return np.nan
        else: # If we have several value, it's stored in a nparray, we take only the first value of this array
            return x[0]

def create_folds(x, y, num_folds, stratified=False, random_state=1):
    """
    Create folds for cross validation.
    
    Parameters
    ----------
    - x : matrix of inputs (array-like)
    - y : vector of labels (array-like)
    - num_folds : number of folds (int)
    - stratified : choose between KFold and StratifiedKFold (bool)
    - random_state : RandomState instance (int)
    """
    
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    return list(folds.split(x, y))

# ----------------------------------------------- Modeling functions --------------------------------------------

def test_dummy_classifiers(x, y, strategies_list=None, random_state=None, constant=None):
    """
    Function which test dummy classifier approaches and return results on some standard metrics.
    
    Parameters
    ----------
    - x : matrix of inputs (array-like)
    - y : vector of labels (array-like)
    - strategies_list : dummy strategies to test (list or None)
    
    Possible values : ['most_frequent', 'prior', 'stratified', 'uniform', 'constant']
    
    If value is None, test all these strategies.
    
    - random_state : RandomState instance (int)
    - constant : The explicit constant as predicted by the "constant" strategy (int or str or array-like)
    """
    
    if not strategies_list:
        strategies_list = ['most_frequent', 'prior', 'stratified', 'uniform', 'constant']

    # Creating a df to store results on tested models
    results_df = pd.DataFrame()
    
    for strategy in strategies_list:
        cst = constant if strategy == 'constant' else None # Must specify a constant if strategy is constant
        
        clf = DummyClassifier(strategy=strategy, random_state=random_state, constant=cst)
          
        # Train dummy and compute time
        start_time = timeit.default_timer()
        clf.fit(x, y)
        fit_time = timeit.default_timer() - start_time

        # Make predictions and compute time
        start_time = timeit.default_timer()
        predictions = clf.predict(x)
        predict_time = timeit.default_timer() - start_time

        probas = clf.predict_proba(x)[:,1]

        # Compute scores
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        roc_auc = roc_auc_score(y, probas)
        cross_entropy = log_loss(y, predictions)
        
        # Store in df
        results_df[strategy] = [accuracy, f1, precision, recall, roc_auc, cross_entropy, fit_time, predict_time]
    
    results_df.index = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'cross_entropy', 'fit_time', 'predict_time']
    return results_df

def quick_classifiers_test(x, y, models_list=None, random_state=None, max_iter=100, n_jobs=None):
    """
    Function which test a bunch of sklearn classification models 
    without hyperparameters optimization or cross-validation and return results on some standard metrics.
    
    Parameters
    ----------
    - x : matrix of inputs (array-like)
    - y : vector of labels (array-like)
    - models_list : models to test (list or None)
    
    Possible values : ['GradientBoostingClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 
    'GaussianProcessClassifier', 'LogisticRegression', 'RidgeClassifier', 
    'SGDClassifier','LinearSVC', 'NuSVC', 'SVC', 'DecisionTreeClassifier']
    
    If value is None, test all these models.
    
    - random_state : RandomState instance (int)
    - max_iter : Maximum number of iterations taken for the solvers to converge (int)
    - n_jobs  : Number of CPU cores used when parallelizing over classes (int)
    """
    
    # If models_list=None, we test all models
    if not models_list:
        models_list = [
            'GradientBoostingClassifier', 
            'RandomForestClassifier', 
            'KNeighborsClassifier',
            'GaussianProcessClassifier', 
            'LogisticRegression', 
            'RidgeClassifier', 
            'SGDClassifier',
            'LinearSVC', 
            'NuSVC', 
            'SVC', 
            'DecisionTreeClassifier'
        ]
    
    # Possible models to test
    models_dict = {
        'GradientBoostingClassifier' : GradientBoostingClassifier(random_state=random_state), 
        'RandomForestClassifier' : RandomForestClassifier(random_state=random_state), 
        'KNeighborsClassifier' : KNeighborsClassifier(n_jobs=n_jobs),
        'GaussianProcessClassifier' : GaussianProcessClassifier(random_state=random_state, max_iter_predict=max_iter, n_jobs=n_jobs), 
        'LogisticRegression' : LogisticRegression(random_state=random_state, max_iter=max_iter, n_jobs=n_jobs), 
        'RidgeClassifier' : RidgeClassifier(random_state=random_state, max_iter=max_iter), 
        'SGDClassifier' : SGDClassifier(random_state=random_state, max_iter=max_iter, n_jobs=n_jobs),
        'LinearSVC' : LinearSVC(random_state=random_state, max_iter=max_iter), 
        'NuSVC' : NuSVC(random_state=random_state), 
        'SVC' : SVC(random_state=random_state), 
        'DecisionTreeClassifier' : DecisionTreeClassifier(random_state=random_state)
    }
    
    # Raise error if unexepected value in models_list
    for model in models_list:
        if model not in models_dict.keys():
            raise ValueError("The model name '{}' isn't in the range of possible values".format(model))
     
    # Classifiy wether the model has predict_proba or decision_function method
    models_probas = [
        'GradientBoostingClassifier', 
        'RandomForestClassifier', 
        'KNeighborsClassifier',
        'GaussianProcessClassifier', 
        'LogisticRegression',
        'DecisionTreeClassifier'
    ]
    
    # Creating a df to store results on tested models
    results_df = pd.DataFrame()
    
    for model in models_list:
        clf = models_dict[model]
            
        # Train model and compute time
        start_time = timeit.default_timer()
        clf.fit(x, y)
        fit_time = timeit.default_timer() - start_time

        # Make predictions and compute time
        start_time = timeit.default_timer()
        predictions = clf.predict(x)
        predict_time = timeit.default_timer() - start_time

        # Predict probas depending on model
        if model in models_probas:
            probas = clf.predict_proba(x)[:,1]
        else:
            probas = clf.decision_function(x)

        # Compute scores
        start_time = timeit.default_timer()
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        roc_auc = roc_auc_score(y, probas)
        cross_entropy = log_loss(y, predictions)
        compute_score_time = timeit.default_timer() - start_time

        # Store in df
        results_df[model] = [accuracy, f1, precision, recall, roc_auc, cross_entropy, fit_time, predict_time, compute_score_time]
    
    results_df.index = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'cross_entropy', 'fit_time', 'predict_time', 'compute_score_time']
    return results_df

def run_GridSearchCV(model, x, y, folds, param_grid, optimized_metric):
    """
    Function which optimizes hyperparameters on a cross-validation (GridSearchCV) for a given model.
    
    Parameters
    ----------
    - model : estimator object
    - x : matrix of inputs (pd.DataFrame, np.array)
    - y : vector of labels (pd.Series, np.array)
    - folds : cross-validation generator or an iterable
    - param_grid : hyperparameters to optimize (dictionnary)
    - optimized_metric : metric to optimize (see sklearn.metrics.SCORERS.keys() for possible values)
    """
    
    clf = GridSearchCV(
        model,
        param_grid,
        cv=folds,
        scoring=optimized_metric,
        n_jobs=-1,
        verbose=4
    )
    
    clf.fit(x, y)
    
    print("Best parameters on training set :")
    print(clf.best_params_)
    print("Best score on training set : {:.3f}".format(clf.best_score_))
    
    return clf
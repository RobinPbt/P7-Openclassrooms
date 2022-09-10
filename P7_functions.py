# Classic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import timeit

# Preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
import imblearn

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
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
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

def test_dummy_classifiers(x, y, valid_size=0.2, strategies_list=None, random_state=None, constant=None, balance_class=False):
    """
    Function which test dummy classifier approaches and return results on some standard metrics.
    
    Parameters
    ----------
    - x : matrix of inputs (array-like)
    - y : vector of labels (array-like)
    - valid_size : proportion of dataset used as validation set (float)
    - strategies_list : dummy strategies to test (list or None)
    
    Possible values : ['most_frequent', 'prior', 'stratified', 'uniform', 'constant']
    
    If value is None, test all these strategies.
    
    - random_state : RandomState instance (int)
    - constant : The explicit constant as predicted by the "constant" strategy (int or str or array-like)
    - balance_class : Decide wheter dataset is over-sampled with SMOTE and under-sampled randomly to balance classes (bool)
    """
    
    if not strategies_list:
        strategies_list = ['most_frequent', 'prior', 'stratified', 'uniform', 'constant']

    # Creating train and valid set
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, random_state=random_state)
        
    # Balance classes
    if balance_class:
        over = imblearn.over_sampling.SMOTE(sampling_strategy=0.1)
        under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        class_balance = imblearn.pipeline.Pipeline(steps=steps)
        x_train, y_train = class_balance.fit_resample(x_train, y_train)
    
    # Creating a df to store results on tested models
    results_df = pd.DataFrame()
    
    for strategy in strategies_list:
        cst = constant if strategy == 'constant' else None # Must specify a constant if strategy is constant
        
        clf = DummyClassifier(strategy=strategy, random_state=random_state, constant=cst)
          
        # Train dummy and compute time
        start_time = timeit.default_timer()
        clf.fit(x_train, y_train)
        fit_time = timeit.default_timer() - start_time

        # Make predictions and compute time
        start_time = timeit.default_timer()
        predictions = clf.predict(x_valid)
        predict_time = timeit.default_timer() - start_time

        probas = clf.predict_proba(x_valid)[:,1]

        # Compute scores
        accuracy = accuracy_score(y_valid, predictions)
        f1 = f1_score(y_valid, predictions)
        precision = precision_score(y_valid, predictions, zero_division=0)
        recall = recall_score(y_valid, predictions)
        roc_auc = roc_auc_score(y_valid, probas)
        cross_entropy = log_loss(y_valid, predictions)
        
        # Store in df
        results_df[strategy] = [accuracy, f1, precision, recall, roc_auc, cross_entropy, fit_time, predict_time]
    
    results_df.index = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'cross_entropy', 'fit_time', 'predict_time']
    return results_df

def quick_classifiers_test(x, y, valid_size=0.2, models_list=None, random_state=None, max_iter=100, n_jobs=None, balance_class=False):
    """
    Function which test a bunch of sklearn classification models 
    without hyperparameters optimization and return results on some standard metrics on a validation set.
    
    
    Parameters
    ----------
    - x : matrix of inputs (array-like)
    - y : vector of labels (array-like)
    - valid_size : proportion of dataset used as validation set (float)
    - models_list : models to test (list or None)
    
    Possible values : ['GradientBoostingClassifier', 'RandomForestClassifier', 'KNeighborsClassifier', 
    'GaussianProcessClassifier', 'LogisticRegression', 'RidgeClassifier', 
    'SGDClassifier','LinearSVC', 'NuSVC', 'SVC', 'DecisionTreeClassifier']
    
    If value is None, test all these models.
    
    - random_state : RandomState instance (int)
    - max_iter : Maximum number of iterations taken for the solvers to converge (int)
    - n_jobs  : Number of CPU cores used when parallelizing over classes (int)
    - balance_class : Decide wheter dataset is over-sampled with SMOTE and under-sampled randomly to balance classes (bool)
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
    
    # Creating train and valid set
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, random_state=random_state)
    
    # Balance classes
    if balance_class:
        over = imblearn.over_sampling.SMOTE(sampling_strategy=0.1)
        under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        class_balance = imblearn.pipeline.Pipeline(steps=steps)
        x_train, y_train = class_balance.fit_resample(x_train, y_train)
    
    # Creating a df to store results on tested models
    results_df = pd.DataFrame()
    
    for model in models_list:
        clf = models_dict[model]
            
        # Train model and compute time
        start_time = timeit.default_timer()
        clf.fit(x_train, y_train)
        fit_time = timeit.default_timer() - start_time

        # Make predictions and compute time
        start_time = timeit.default_timer()
        predictions = clf.predict(x_valid)
        predict_time = timeit.default_timer() - start_time

        # Predict probas depending on model
        if model in models_probas:
            probas = clf.predict_proba(x_valid)[:,1]
        else:
            probas = clf.decision_function(x_valid)

        # Compute scores
        start_time = timeit.default_timer()
        accuracy = accuracy_score(y_valid, predictions)
        f1 = f1_score(y_valid, predictions)
        precision = precision_score(y_valid, predictions)
        recall = recall_score(y_valid, predictions)
        roc_auc = roc_auc_score(y_valid, probas)
        cross_entropy = log_loss(y_valid, predictions)
        compute_score_time = timeit.default_timer() - start_time

        # Store in df
        results_df[model] = [accuracy, f1, precision, recall, roc_auc, cross_entropy, fit_time, predict_time, compute_score_time]
    
    results_df.index = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'cross_entropy', 'fit_time', 'predict_time', 'compute_score_time']
    return results_df

def run_GridSearchCV(model, x, y, folds, param_grid, optimized_metric, balance_class=False):
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
    - balance_class : Decide wheter dataset is over-sampled with SMOTE and under-sampled randomly to balance classes (bool)
    """
    
    # Balance classes
    if balance_class:
        over = imblearn.over_sampling.SMOTE(sampling_strategy=0.1)
        under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=0.5)
        
        # We create a imblearn pipeline which will take care to balance only train set during cross validation
        steps = [('o', over), ('u', under), ('model', model)]
        pipe = imblearn.pipeline.Pipeline(steps=steps)
        
        # We have to rename the parameters accordingly to pipeline step
        params = {}
        for param, value in param_grid.items():
            params['model__{}'.format(param)] = value
        
    else:
        pipe = model
        params = param_grid

    
    # Proceeding to cross-validation with parameters optmization
    clf = GridSearchCV(
        pipe,
        params,
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

def test_classification_thresholds(model, x, y, threshold_list = np.linspace(0.05, 0.5, num=10)):
    """
    Function which computes classifications metrics for a model with different prediction thresholds.
    
    Parameters
    ----------
    - model : estimator object
    - x : matrix of inputs (pd.DataFrame, np.array)
    - y : vector of labels (pd.Series, np.array)
    """
    
    test_df = pd.DataFrame()
    
    probas = model.predict_proba(x)[:,1]
    threshold_list = np.linspace(0.05, 0.5, num=10)

    for i in threshold_list:
        y_pred = (probas >= i).astype(int)
        fbeta = fbeta_score(y, y_pred, beta=2)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        scores = [fbeta, accuracy, precision, recall]
        test_df[i] = scores
        
    test_df.index = ['fbeta', 'accuracy', 'precision', 'recall']
    return test_df
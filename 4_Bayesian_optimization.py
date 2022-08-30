
import numpy as np
from sklearn import  ensemble  
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing, decomposition, pipeline

from skopt import space
from skopt import gp_minimize
import pandas as pd

from functools import partial


def  optimize(params, params_name, x, y):
    params = dict(zip(params_name, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracy = []
    for idx in kf.split(X=x, y=y):
        train_idx , test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
    
        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracy.append(fold_acc)

    return -1.0 * np.mean(accuracy)
    

if __name__ == "__main__":
    df = pd.read_csv("./data/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    param_space = [
        space.Integer(3,15,name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, name="max_features")
    ]

    params_names = ["max_depth", "n_estimators", "criterion", "max_features"]


    optimization_fun = partial(optimize, params_name=params_names, x=X, y=y)
    results = gp_minimize(
        optimization_fun, dimensions=param_space, n_calls = 15, n_random_starts=10, verbose=10
    )

    print(dict(zip(params_names, results.x)))
    #{'max_depth': 15, 'n_estimators': 561, 'criterion': 'entropy', 'max_features': 1.0}
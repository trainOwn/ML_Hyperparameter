from tabnanny import verbose
import numpy as np
from sklearn import  ensemble  
from sklearn import metrics
from sklearn import model_selection
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("./data/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100,300,500],
        "max_depth" : np.arange(1,20),
        "criterion": ["gini", "entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator = classifier,
        param_distributions = param_grid,
        scoring = "accuracy",
        n_iter = 10,
        n_jobs = 1 ,
        cv = 5,
    )

    model.fit(X,y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
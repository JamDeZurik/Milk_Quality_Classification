import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

#runs the random forest grid search
def run_random_forest(X_train, X_test, y_train, y_test):
    #parameters for a grid search (ranges to test)
    param_grid_rf = {
        'max_depth': range(3, 5),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid_rf, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    max_depth = grid_search.best_params_["max_depth"]
    min_samples_split = grid_search.best_params_["min_samples_split"]
    min_samples_leaf = grid_search.best_params_["min_samples_leaf"]
    print(f"Best Max_Depth: {max_depth}, Best Sample_Split: {min_samples_split}, Best Sample_Leaf: {min_samples_leaf}")


    clf = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, oob_score=True)
    clf.fit(X_train, y_train)

    importances = pd.DataFrame(clf.feature_importances_, index=df.columns[0:-1])
    importances.plot.bar()
    plt.show()

    print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
    print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
    print(f"OOB Score: {clf.oob_score_:.3f}")

    display_confusion_matrix(clf,y_test,X_test)

#runs the svc grid search
def run_linear_svc(X_train, X_test, y_train, y_test):
    param_grid_svc = {
        'base_estimator__C': [1, 10, 100],
        'base_estimator__gamma': np.linspace(0.01, 0.1, 10)
    }

    svc = SVC(kernel='linear', class_weight='balanced') #imbalanced classes
    bag = BaggingClassifier(svc, max_features=0.25, n_estimators=100, verbose=3, oob_score=True, n_jobs=-1)

    grid_search = GridSearchCV(estimator=bag, param_grid=param_grid_svc, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)
    C = grid_search.best_params_["base_estimator__C"]
    gamma = grid_search.best_params_["base_estimator__gamma"]

    svc = SVC(kernel='linear', C=C, gamma=gamma)
    clf = BaggingClassifier(svc, max_features=0.25, n_estimators=100, verbose=3, oob_score=True, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
    print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
    print(f"OOB Score: {clf.oob_score_:.3f}")

    display_confusion_matrix(clf, y_test, X_test)

#display a confusion matrix of the results
def display_confusion_matrix(clf, y_test, X_test):
    #genres with indices corresponding to "class" column

    cm = confusion_matrix(y_test, clf.predict(X_test), normalize='true')
    disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_.tolist())
    disp_cm.plot()
    plt.show()

# Milk Quality
# https://www.kaggle.com/datasets/yrohit199/milk-quality
df = pd.read_csv("milknew.csv")

# Select variables
y = df.iloc[:, -1].copy().to_numpy() #Genre (12 Classes)
X = df.iloc[:, 0:-1].copy().to_numpy()

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

run_random_forest(X_train,X_test,y_train,y_test)
run_linear_svc(X_train,X_test,y_train,y_test)

# Decision trees are able to separate the data from highest to lowest priority
# important for this dataset since pH and temperature weighted higher than others
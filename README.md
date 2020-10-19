# KAGGLE ENSEMBLING GUIDE
ref: https://mlwave.com/kaggle-ensembling-guide/



### Model ensembling is a very powerful technique to increase accuracy on a variety of ML tasks. In this article I will share my ensembling approaches for Kaggle Competitions.

## 1. Creating ensembles from submission files

### 1.1 Voting ensembles.
- Error correcting codes

- A machine learning example

- A pinch of maths

- Number of voters

- Correlation

- Use for Kaggle: Forest Cover Type prediction

- Weighing

- Use for Kaggle: CIFAR-10 Object detection in images

- Code

### 1.2 Averaging
- Kaggle use: Bag of Words Meets Bags of Popcorn

- Code

### 1.3 Rank averaging
- Historical ranks.

- Kaggle use case: Acquire Valued Shoppers Challenge

- Code






## 2. Stacked Generalization & Blending

### Averaging prediction files is nice and easy, but it’s not the only method that the top Kagglers are using. The serious gains start with stacking and blending. Hold on to your top-hats and petticoats: Here be dragons. With 7 heads. Standing on top of 30 other dragons.

### 2.1 Netflix

### 2.2 Stacked generalization

### 2.3 Blending


## 3. Why create these Frankenstein ensembles?


# Overfitting in machine learning
ref: https://elitedatascience.com/overfitting-in-machine-learning

### 1. Examples of Overfitting

### 2. Signal vs. Noise

### 3. Goodness of Fit

### 4. Overfitting vs. Underfitting

### 5. How to Detect Overfitting

- If our model does much better on the training set than on the test set, then we’re likely overfitting.

### 6. How to Prevent Overfitting

### 7. Additional Resources

# Python machine learning tutorial, Scikit-Learn
ref: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

Here are the steps for building your first random forest model using Scikit-Learn:

### 1. Set up your environment.

### 2. Import libraries and modules.

### 3. Load red wine data.

### 4. Split data into training and test sets.

### 5. Declare data preprocessing steps.

### 6. Declare hyperparameters to tune.

### 7. Tune model using cross-validation pipeline.

### 8. Refit on the entire training set.

### 9. Evaluate model pipeline on test data.

### 10. Save model for further use.

     # 2. Import libraries and modules
     import numpy as np
     import pandas as pd
 
     from sklearn.model_selection import train_test_split
     from sklearn import preprocessing
     from sklearn.ensemble import RandomForestRegressor
     from sklearn.pipeline import make_pipeline
     from sklearn.model_selection import GridSearchCV
     from sklearn.metrics import mean_squared_error, r2_score
     from sklearn.externals import joblib 
 
     # 3. Load red wine data.
     dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
     data = pd.read_csv(dataset_url, sep=';')
 
     # 4. Split data into training and test sets
     y = data.quality
     X = data.drop('quality', axis=1)
     X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                         test_size=0.2, 
                                                        random_state=123, 
                                                        stratify=y)
 
     # 5. Declare data preprocessing steps
     pipeline = make_pipeline(preprocessing.StandardScaler(), 
                              RandomForestRegressor(n_estimators=100))
 
     # 6. Declare hyperparameters to tune
     hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                       'randomforestregressor__max_depth': [None, 5, 3, 1]}
 
     # 7. Tune model using cross-validation pipeline
     clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
     clf.fit(X_train, y_train)
 
     # 8. Refit on the entire training set
     # No additional code needed if clf.refit == True (default is True)
 
     # 9. Evaluate model pipeline on test data
     pred = clf.predict(X_test)
     print r2_score(y_test, pred)
     print mean_squared_error(y_test, pred)
 
     # 10. Save model for future use
     joblib.dump(clf, 'rf_regressor.pkl')
     # To load: clf2 = joblib.load('rf_regressor.pkl')
     
     

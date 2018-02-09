#!/usr/bin/env python

# Kris Pickrell

# Description:  This is a simple example demonstrating a robust technique for ensembling several multilabel 
# classifiers that can help solve both the overfitting problem and the meta-parameter estimation search
# problem.  The intuition that guides this solution is two-fold.  First, it is highly likely that for any
# two estimators, the performance will vary between example classes in a complimentary way, that is, one 
# will do well where the other does poorly and vice versa.  Second, since both estimators are likely trained on
# the same data and differ only by meta-parameters, the performance will also likely be correlated.  That is,
# each should do better than random and probably in the same general direction, hence correlated.  Such correlation
# injects a problem of collinearity into any ensembling technique that could occur.  Collinearity tends to cause many
# problems, one of which is the inflation of the coefficients, and consequently the variance of the estimator.  Hence,the need for
# Tikhonov regularization (aka ridge regression, L2 penalty) in the solution, which seeks to minimize not only the 
# objective loss function, but the weighted sum of squared parameters (not the bias, just the weights).  It can be 
# shown than the Tikhonov regularized solution is equivalent to the Bayesian MAP estimator which is a linear interpolation between the
# prior mean (sigmoid distribution of any single estimator used in the ensemble) and the covariance-weighted sample mean:
# the ensemble estimate.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from scipy.stats.distributions import chi2

import keras
from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras import backend as K

from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from keras.wrappers.scikit_learn import KerasClassifier
from functools import partial

from sklearn.linear_model import LogisticRegression

def _main():
    # Pick an arbitrary number of labels
    n_classes = 10

    # Number of samples to generate
    n_samples = 10000
    X, y = make_multilabel_classification(n_samples=n_samples,
                                          n_features=50,
                                          n_classes=n_classes,
                                          n_labels=5,
                                          length=50,
                                          allow_unlabeled=False,
                                          sparse=False,
                                          return_indicator='dense',
                                          return_distributions=False,
                                          random_state=None)

    # X is (10000,50), y is (10000,10) with an average of 5 labels per sample

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # Just dump the correlation matrix to confirm that there is some inter-label correlation
    # along with the t-statistics and chi2 values to show that it is not insignificant
    df = pd.DataFrame(y)
    print( 'n_samples={}, corr'.format(n_samples) )
    tvalue1 = 0.05/np.sqrt( (1-0.05**2)/(n_samples-2) )
    tvalue2 = 0.1/np.sqrt( (1-0.1**2)/(n_samples-2) )
    print( '{}/sqrt( (1-{}**2)/(n_samples-2) ) == {}'.format(0.05, 0.05, tvalue1) )
    print( '{}/sqrt( (1-{}**2)/(n_samples-2) ) == {}'.format(0.1, 0.1, tvalue2) )
    print( 'X-squared value: {}, {}'.format( 0.05, chi2.cdf(tvalue1,2) ) )
    print( 'X-squared value: {}, {}'.format( 0.1, chi2.cdf(tvalue2,2) ) )
    print( df.corr() )

    input_dim = X_train.shape[1]
    
    # First, build some hand-tooled network, obviously guessing at the optimal meta-parameters

    print( 'hand-tooling single fully-connected network...' )
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.1))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_classes, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, nesterov=True, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    model.fit(X_train, y_train, epochs=5, batch_size=int(n_samples*0.2))

    predictions = model.predict(X_test)

    print( classification_report(y_test, predictions.round()) )

    # Now, specify some random, also non-optimal ranges to play with and create several
    # models within this space.  In the end, we will combine the predictions from each
    # of them in two-stages.  1.  Output n_ensembles predictions for each of the n class
    # labels.  These predictions train a ridge on it's particular class label.  2. Combine
    # all of the ridge predictions using another ridge and produce the predictions.

    n_ensembles = 100
    print( 'ridge regression of {} semi-random estimators...'.format(n_ensembles) )
    models = []
    for _ in xrange(n_ensembles):
        estimator_type = random.choice(['neuralnet'])
        
        if estimator_type == 'neuralnet':
            model = Sequential()
            for i in xrange(np.random.randint(2,8)):
                n_neurons = np.random.randint(400,1200)
                n_dropout = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                model.add(Dense(n_neurons, activation='relu', input_dim=input_dim))
                model.add(Dropout(n_dropout))
    
            model.add(Dense(n_classes, activation='sigmoid'))
        
            sgd = SGD(lr=random.choice([0.01,0.02]), decay=1e-6, nesterov=random.choice([True,False]), momentum=random.choice([0.8,0.9]))
            model.compile(loss='binary_crossentropy', optimizer=sgd)
        
            model.fit(X_train, y_train, epochs=np.random.randint(5,8), batch_size=int(n_samples*random.choice([0.1,0.2,0.3])))
        elif estimator_type == 'forest':
            model = RandomForestClassifier( n_estimators=random.choice(range(300,600,50)) )
            model.fit(X_train, y_train)

        models.append(model)

    def ridgeit(X, models):
        ridge_predictions = np.array([])
        for model in models:
            X_ridge = model.predict(X)
            if len(ridge_predictions):
                ridge_predictions = np.hstack((ridge_predictions,X_ridge))
            else:
                ridge_predictions = X_ridge
        return ridge_predictions

    print( 'building ridges...' )
    ridge = { i : { model:LogisticRegression(penalty='l2') for model in models } for i in xrange(n_classes) }
    ensemble = { i : LogisticRegression(penalty='l2') for i in xrange(n_classes) }
    for label in xrange(n_classes):
        y_ridge = y_train[:,label]
        for model in models:
            X_ridge = model.predict(X_train)
            ridge[label][model].fit(X_ridge, y_ridge)

        ridge_predictions = ridgeit(X_train, ridge[label])
        ensemble[label].fit(ridge_predictions,y_ridge)

    print( 'making predictions...' )
    y_predictions = np.zeros(y_test.shape)
    for label in xrange(n_classes):
        ridge_predictions = ridgeit(X_test, ridge[label])
        predictions = ensemble[label].predict(ridge_predictions)
        y_predictions[:,label] = predictions
    
    print( classification_report(y_test,y_predictions) )

    return 0

if __name__ == '__main__':
    import sys;
    sys.exit( _main() )


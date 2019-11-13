# Basiq ML Challenge

This repository contains the code that performs bank transaction classification over the training set of 100 000 observations. The code demonstrates the approach that consists of the following steps:
- preprocessing the data ([data_preparation.py](data_preparation.py),
- building classification models ([naive_bayes.py](naive_bayes.py)) and [svc.py](svc.py)), and
- making predictions over the scorecard data with the best performing classifier ([fill_scorecard.py](fill_scorecard.py)). 

The algorithms chosen are Naive Bayes and SVM (with the linear kernel); both known to perform well with text classification problems, but yet simple to train without requiring high computational time and resources (also reasons for choosing them).
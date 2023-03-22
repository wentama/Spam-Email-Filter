# Spam-Email-Filter

A model that classifies emails as spam/non-spam. This case study is an introductory to supervised machine learning and served as an opportunity to play around with different classifiers and understand how they work to choose the best one for the given dataset. 

## Evaluation 

Round 1 Evaluation: 'spamTrain1.csv' is the training data. 'spamTrain2.csv' is the testing data. This is evaulated using 'evaluateClassifier.py'

Final Evaluation: 'spamTrain1.csv' + 'spamTrain2.csv' is the training data. 'spamTest.csv' is the testing data. This is evaulated using 'evaluateClassifierTest.py'

The final model has a 0.9282 AUC and 0.5609 TPR at FPR=0.01 on the unknown testset 'spamTest.csv'

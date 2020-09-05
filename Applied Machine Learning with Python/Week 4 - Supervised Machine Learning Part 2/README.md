## WEEK 3 - EVALUATION

### Dummy Classifier
***Dummy Classifiers or Regressors completely ignore input data.***<br>
DummyClassifier is a classifier that makes predictions using simple rules.
```python
from sklearn.dummy import DummyClassifier

# For eg if Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0
y_dummy_predictions = dummy_majority.predict(X_test)
```
* `strategy` param here is used to specify the strategy we want the classifier to follow while giving prediction.
  - `most_frequent` simply assigns all predictions to the most occurring class in the input data
  - `stratified` gives predictions according to the input data's class distribution(randomly)
  - `uniform` predicts uniformly at random
  - `constant` lets the user give the prediction class

### Evaluation Metrics
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
```python
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_majority_predicted)
print(confusion)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))
```

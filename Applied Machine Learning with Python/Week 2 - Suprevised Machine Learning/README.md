## WEKK 2 - SUPREVISED MACHINE LEARNING

### kNN
- ***kNN is mostly good for data having less number of features, more number of features may slow down k nearest neighbor algorithm. So better use kNN in your ML model when input data has less number of features.***
- Also as k increases the algorithm tries to find more generalized patterns in the training set but there is an upper limit at which the prediction accuracy of kNN algorithm reaches to a peak then starts decreasing if k is increased further.
- Basically for **k = 1 mostly the model overfits the training data** by making too complex patterns and thus not giving accurate predictions for the unseen data(test data) since it does not recognizes more generalized patterns.
- On the other hand for **very large k**(like equal to half of number of training instances) **the model underfits the training data** by making too simple patterns and thus not even giving good accuracy with training data itself since the model is so ***noob*** to find any patterns in the data.
- In **kNN for classification problems** the algorithm finds k nearest neighbors to the test data point and then simply takes majority vote to decide the final class of the test data point.
- In **kNN for regression problems** what the algorithm does is that it first do the same step that finds k nearest neighbors to the test data point and then since there is no discrete class so instead of taking any majority vote, it takes average of all the distances between test data point and each one of the k points to decide the prediction.

### Least-Squares Linear Regression
- Linear Regression is an example of Linear Model.<br>
- **Basic equation to represent linear regression is the simple straight line equation in mathematics i.e.** ***y = mx + c*** which is moreover recognized as ***y = wX + b*** where<br>
y is the output labels/predictions vector,<br>
X is the feature vector,<br>
w is the model coefficients/feature weights vector,<br>
b is bias term or the intercept of the model<br>
- To find the weight w and bias term b linear regression uses a very common method of **Ordinary Least Squares** which finds the line through the training set which minimizes the *mean squared error(loss function)* of the model.<br>
(The **mean squared error** is essentially the sum of the squared differences between the predicted target value & the actual target value for all the points in the training set)
- ***Linear models may seem simplistic but for data with many features linear models can be very effective and generalize well to new data beyond the training set.***
- In a linear model there are no parameters to control model complexity, linear model always uses all of the input variables and always is represented by a straight line.
```python
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)
linreg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'
     .format(linreg.coef_))
print('linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, y_test)))
```
- `coef_` attribute stores the feature weights w of the linear model and is called coefficients of the model.
- `intercept_` attribute stores the bias term b of the model<br>
(If a scikit learn object attribute ends with an underscore, this means that the attribute were derived from training data and were quantities set by user)<br>
- `fit_intercept` param is a boolean to determine whether to calculate the intercept of the model which is True by default
- `normalize` param is also a boolean to, which is False by default & if True, the features `X` will be normailzed before regression by subtracting mean and dividing by L2-norm(The L2 norm calculates the distance of the vector coordinate from the origin of the vector space, it is also called euclidean norm as the distance is same as euclidean distance).<br>
(If you wish to normalize, use `sklearn.preprocessing.StandardScaler` before applying `fit` on an estimator with `normalize = False`)
- `n_jobs` param is to specify the number of processors to be used for calculation, **it will only provide speedup if number of labels/targets > 1 & sufficient large data.** -1 means using all processors.

### k-NN Regression vs Least-Squares Linear Regression
- The k nearest neighbor regressor doesn't make a lot of assumptions about the data and gives potentially accurate but sometimes unstable predictions that are sensitive to small changes in the training data.
- On the other hand linear models make strong assumptions about the structure of the data, in other words, the target value can be predicted using the weighted sum of the input variables. And linear models give stable but inaccurate predictions.

### Ridge Regression/Tikhonov Regularization
It is same as linear regression but with one difference, during the training phase it adds a penalty for feature weights that are too large. Basically it is least squares with L2 regularisation.
```python
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
linridge = Ridge(alpha=20.0).fit(X_train, y_train)
```
- `alpha` param decides the strength of regularisation to be applied
- `fit_intercept` and `normalize` params same as in linear regression
- `coef_` and `intercept_` attributes same as in linear regression

**Regularisation**
- The addition of penalty term to a learning algorithm's objective function is called Regularisation.
- It's a way to prevent overfitting, and, thus improve the generalization performance of a model by restricting the model's possible parameter settings. Usually the aim of restriction in regularisation is to reduce the complexity of final estimated model.
- The amount of regularisation to be applied is controlled by alpha parameter. Larger alpha means more regularisation & simpler linear models with shrinking the weights towards to zero & towards each other.
- Alpha parameter is also called ridge regression regularisation penalty or L2 penalty.<br>

**Feature Preprocessing & Normalization**<br>
If the input variables, the features have very different scales, then when this shrinkage(regularisation or L2 penalty) of the coefficients happens, input variables with different scales will have different contributions to this L2 penalty, because the L2 penalty is the sum of squares of all the coefficients. So transforming the input features, so they're all on the same scales, means the ridge penalty is in some sense applied more fairly to all features.

- ***Beyond just regularized regression, normalization is important to perform for a number of different machine learning algorithms.***
- The type of feature preprocessing & normalization that's needed can also depend on the data.
- One of the more widely used form of feature normalization called **MinMax Scaling**. It is applied to all the features by:<br>
X<sub>i</sub> = (X<sub>i</sub> - X<sub>i</sub><sup>MIN</sup>) / (X<sub>i</sub><sup>MAX</sup> - X<sub>i</sub><sup>MIN</sup>)
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

# fit and transform in single step by fit_transform() method
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
```
- `feature_range` param decides the min-max interval for the transformed data, default is (0,1)

**Critical Aspects of Feature Normalization**
- First, that we're applying same scaler object to both training and testing data.
- Second, that we're training the scaler object on the training data and not on the testing data.
- If you don't apply same scaling to both training and test sets, you'll end up more or less with random **DATA SKEW**(Data skew primarily refers to a non uniform distribution in a dataset), which will invalidate your result.
- If you prepare the scaler or any other normalization method by showing it the test data instead of training data, this leads to a phenomenon called **DATA LEAKAGE** where the training phase has information that is leaked from the test set which the learner should never have access to during training.
- One downside to performing feature normalization is that the resulting model and the transformed features may be harder to interpret.
- In the end, the type feature normalization that's best to apply, can depend on the dataset, learning task & learning algorithm to be used.
- ***Regularisation becomes less important as the amount of training data increases.***

### Lasso Regression
It is similar to ridge regression, just uses a different L1 penalty instead of L2 penalty. It applies the penalty as sum of the absolute values of the w coefficients instead of sum of squared values of the w coefficients. This results in making a subset of feature coefficients forcibly equal to zero, and mostly the most important features are left with the non-zero weights.<br>
The lasso regression results do help us see some of the strongest relationships between the input variables & outcomes for a particular data.

### Polynomial Regression
- It is similar to linear regression, just that it adds polynomial features to the input features which are basically all possible combinations between the features.
- The advantage of using these extra polynomial features is that the model trained by polynomial features can detect non-linear relationships within the data which cannot be done by simple linear model.
- But the disadvantage of overfitting also comes with this as the with the polynomial features the model becomes more complex, and thus, may fail to detect the more generalized patterns for unseen data. So in practice, polynomial regression is often done with a regularised learning method like ridge regression.
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ridge

poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state = 0)
```
- `degree` param specifies degree for the polynomial features, default is 2
- `n_input_features_` attribute gives the number of input features given
- `n_output_features_` attribute gives the total number of output features after applying polynomial regression

# LEARNINGS FROM APPLIED MACHINE LEARNING WITH PYTHON

## WEEK 1 - FUNDAMENTALS OF MACHINE LEARNING

### count values for different labels
```python
# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
lookup_fruit_name
```
The above code comes in very handy for dataset having categories and labels(value specifying a specific category) to get a very brief idea of distribution of data for various categories in the dataset.

### pandas value_counts()
Similar to above method `df['column'].value_counts()` is really useful function to overview the number of unique values in a column of a dataframe. It gives series as its output with index as labels and count of those labels as column of the series.

### matplotlib colormap
Now there is something called colormap (also called cmap), for various graphs and plot we can use many different colormaps having variety of color patterns.
```python
from matplotlib import cm
# get all colormaps you can use
cm.cmap_d.keys()
# example cmap
cmap = cm.get_cmap('rainbow')
```

### pandas scatter_matrix
**pandas scatter matrix is simply as its name follows, a matrix of scatter plots.**<br>
For using pandas scatter_matrix you have to import it separately or simply using `pd.scatter_matrix()` will throw an error or else use `pd.plotting.scatter_matrix()`.
```python
from pandas.plotting import scatter_matrix
```
Basically pandas scatter_matrix is used to get insights of relationships between multiple different columns(features) by plotting scatter plots between them at the same time.
```python
scatter = scatter_matrix(X_train, c=y_train, marker = 'o', hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
```
- The first arg is data for which we have to plot scatter matrix
- `c` is for telling scatter_matrix that each unique category of data should be identified uniquely(by color)
- `cmap`(mentioned above) is just for getting different color combinations in the scatter plots of the matrix
- `diagonal` which can take two values: {hist, kde} i.e. the diagonal of the scatter_matrix gives idea about distribution of values in individual columns(features)
- For these two values of `diagonal` pandas also gives an option to specify keyword arguments as `hist_kwds` & `density_kwds` for Histogram or KDE(Kernel Density Estimation) plot respectively by which one can get desired things in the diagonal plots of the scatter_matrix
- `s` can also be used as an arg for specifying the marker size

### 3D Scatter Plot
```python
# plotting a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()
```
- It is as simple as it looks, the first three args are the x, y & z axes
- `c` is same as for scatter_matrix to tell scatter plot that each category should be identified uniquely(by color)
- `s` is to specify the marker size, that how big a point on the scatter plot should look

## WEEK 2 - SUPERVISED MACHINE LEARNING

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

### k-NN Regression vs Least-Squares Linear Regression
- The k nearest neighbor regressor doesn't make a lot of assumptions about the data and gives potentially accurate but sometimes unstable predictions that are sensitive to small changes in the training data.
- On the other hand linear models make strong assumptions about the structure of the data, in other words, the target value can be predicted using the weighted sum of the input variables. And linear models give stable but inaccurate predictions.

### Ridge Regression/Tikhonov Regularization
It is same as linear regression but with one difference, during the training phase it adds a penalty for feature weights that are too large. Basically it is least squares with L2 regularisation.

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

### Logistic Regression
Everything same as linear regression but with a difference in the function to compute weights & the bias term i.e. it uses a special non-linear logistic function(1 / (1 + exp<sup>-(b + wX)</sup>)) instead of a straight line. In logistic regression the regularisation is controlled by parameter `C` instead of `alpha` which is generally used for regularisation.

### Kernelized SVM or SVM(Support Vector Machine)
There are problems where a linear model having line or hyperplane cannot classify the data well. For these types of problems the use of **Kernelized SVM** comes into the picture which is **powerful extension of linear support vector machines**.
- As like other supervised learning methods SVM can be used for both classification & regression.
- What Kernelized SVM do is that it takes the original data space & transform it into a new higher dimensional feature space where it becomes much easier to classify the transformed data using linear classifier.
- There are various kernels available for the Kernelized SVM which corresponds to different transformations. For eg, Radial Basis Function Kernel(RBF), Polynomial Kernel, etc.
- The kernel function in SVM tells us, given two points in the original input space, what is their similarity in the new feature space.
- For the RBF kernel function the similarity between two points and the transformed feature space is an exponentially decaying function of distance between the vectors and the original input space as shown by the formula here.(K(x, x<sup>'</sup>) = e<sup>-(gamma * (x - x<sup>'</sup>)<sup>2</sup>)</sup>)
- `gamma` controls how far the influence of a single trending example reaches, which in turn effects how tightly the decision boundaries end up surrounding points in the input space.
  * Small `gamma` means a large similarity radius, so the points farther apart are considered similar which results in more points being grouped together & smoother decision boundaries.
  * For larger `gamma` the kernel value decays more quickly and points have to be very close to be considered similar which results in more complex, tightly constrained decision boundaries.
- SVM also has a regularisation parameter `C`, that controls the trade off between satisfying the maximum margin criterion to find the simple decision boundary, and avoiding misclassification errors on the training set.
  * Smaller `C` or by reducing `C` will get you more smoother decision boundaries.
  * Larger `C` or by increasing `C` will get you more tightly constrained boundaries.
- Typically `C` & `gamma` interacts with each other & hence tuned together.
  * If `gamma` is large, `C` will have middle to low effect.
  * If `gamma` is small, the model is much more constrained and the effect of `C` would be similar to how it would affect a linear classifier.

## Scikit Learn
- ***train_test_split()***
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  ```
  * What `train_test_split()` does is that it splits the data into training & testing datasets with default ratio of 3:1
  * There are args named `train_size` & `test_size` to specifically specify them if you want
  * The first two args are features and labels of the data respectively
  * Then there is an argument `random_state` which is as obvious by name a random number, what it does is that it controls the shuffling of data before applying the split i.e. **you can always get the same result by keeping the value unchanged for** `random_state`

  (Here the conventional notation is used i.e. X(capital x) is used to represent features of the data & y(small y) is used to represent labels of the data)

- ***KNeighborsClassifier()***
  ```python
  # create classifier object
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(*args)
  ```
  * `KNeighborsClassifier()` comes in very handy when it comes to applying k nearest neighbor algorithm, it can be used for both regression & classification.<br>
  * `n_neighbors` specifies number of neighbors used by kNN classifier to predict a result i.e. when kNN takes input for predicting it calculates the distance(euclidean distance is default in scikit learn) of that input data point with its k(i.e. n_neighbors) nearest data points & then on basis of `algorithm` arg(which is taking majority by default) between those points it gives the prediction.
  ```python
  # train the classifier(fit the estimator) using training data
  knn.fit(X_train, y_train)
  # estimate accuracy on the future data using the test data
  knn.score(X_test, y_test)
  # use the trained k-NN classifier model to classify new, previously unseen objects
  # example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm
  fruit_prediction = knn.predict([[20, 4.3, 5.5]])
  lookup_fruit_name[fruit_prediction[0]]
  ```

- ***LinearRegression()***
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
  * `coef_` attribute stores the feature weights w of the linear model and is called coefficients of the model.
  * `intercept_` attribute stores the bias term b of the model<br>
  (If a scikit learn object attribute ends with an underscore, this means that the attribute were derived from training data and were quantities set by user)<br>
  * `fit_intercept` param is a boolean to determine whether to calculate the intercept of the model which is True by default
  * `normalize` param is also a boolean to, which is False by default & if True, the features `X` will be normailzed before regression by subtracting mean and dividing by L2-norm(The L2 norm calculates the distance of the vector coordinate from the origin of the vector space, it is also called euclidean norm as the distance is same as euclidean distance).<br>
  (If you wish to normalize, use `sklearn.preprocessing.StandardScaler` before applying `fit` on an estimator with `normalize = False`)
  * `n_jobs` param is to specify the number of processors to be used for calculation, **it will only provide speedup if number of labels/targets > 1 & sufficient large data.** -1 means using all processors.

- ***Ridge()***
  ```python
  from sklearn.linear_model import Ridge
  X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
  linridge = Ridge(alpha=20.0).fit(X_train, y_train)
  ```
  * `alpha` param decides the strength of regularisation to be applied
  * `fit_intercept` and `normalize` params same as in linear regression
  * `coef_` and `intercept_` attributes same as in linear regression

- ***MinMaxScaler()***
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
  * `feature_range` param decides the min-max interval for the transformed data, default is (0,1)

- ***Lasso()***
  ```python
  from sklearn.linear_model import Lasso
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()

  X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)

  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)
  ```
  * `max_iter` param is to avoid warning which may occur for some datasets, to avoid those set it 20,000 or further

- ***PolynomialFeatures()***
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.linear_model import ridge

  poly = PolynomialFeatures(degree = 2)
  X_poly = poly.fit_transform(X)

  X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state = 0)
  ```
  * `degree` param specifies degree for the polynomial features, default is 2
  * `n_input_features_` attribute gives the number of input features given
  * `n_output_features_` attribute gives the total number of output features after applying polynomial regression

- ***LogisticRegression()***
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression

  X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
  clf = LogisticRegression(C = 10).fit(X_train, y_train)
  ```
  * `C` param is used to control the amount of regularisation to be applied, larger value of `C` defines less regularisation, its default value is 1

- ***SVC()***
  ```python
  from sklearn.svm import SVC
  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
  clf = SVC(kernel = 'rbf', gamma=5.0).fit(X_train, y_train)
  ```
  * `kernel` param is used to decide the type of kernel function to be used, the advantage with this param is that it allows to set different types of kernels including customized functions. Default is RBF function.
  * `gamma` param decides the kernel width, and is a very sensitive param for SVM.
  * `C` is the regularisation parameter which is tuned with `gamma` for optimized performance.
  
- ***DummyClassifier()***
  ```python
  from sklearn.dummy import DummyClassifier

  # For eg if Negative class (0) is most frequent
  dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
  # Therefore the dummy 'most_frequent' classifier always predicts class 0
  y_dummy_predictions = dummy_majority.predict(X_test)
  ```
  ***Dummy Classifiers or Regressors completely ignore input data.***<br>
  DummyClassifier is a classifier that makes predictions using simple rules.
  * `strategy` param here is used to specify the strategy we want the classifier to follow while giving prediction.
    - `most_frequent` simply assigns all predictions to the most occurring class in the input data
    - `stratified` gives predictions according to the input data's class distribution(randomly)
    - `uniform` predicts uniformly at random
    - `constant` lets the user give the prediction class
    
 - ***sklearn.metrics***
   - Confusion Matrix
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC Curve
   ```python
   from sklearn.metrics import confusion_matrix
   confusion = confusion_matrix(y_test, y_majority_predicted)
   print(confusion)

   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
   # Accuracy = TP + TN / (TP + TN + FP + FN)
   # Precision = TP / (TP + FP)
   # Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
   # F1 = 2 * Precision * Recall / (Precision + Recall)
   print('Accuracy: {:.2f}'.format(accuracy_score(y_test, tree_predicted)))
   print('Precision: {:.2f}'.format(precision_score(y_test, tree_predicted)))
   print('Recall: {:.2f}'.format(recall_score(y_test, tree_predicted)))
   print('F1: {:.2f}'.format(f1_score(y_test, tree_predicted)))


## WEEK 1: FUNDAMENTAL OF MACHINE LEARNING - Intro to Scikit Learn

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

### Scikit Learn
- ***train_test_split()***
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```
- What `train_test_split()` does is that it splits the data into training & testing datasets with default ratio of 3:1
- There are args named `train_size` & `test_size` to specifically specify them if you want
- The first two args are features and labels of the data respectively
- Then there is an argument `random_state` which is as obvious by name a random number, what it does is that it controls the shuffling of data before applying the split i.e. **you can always get the same result by keeping the value unchanged for** `random_state`

(Here the conventional notation is used i.e. X(capital x) is used to represent features of the data & y(small y) is used to represent labels of the data)

- ***KNeighborsClassifier()***
```python
# create classifier object
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(*args)
```
`KNeighborsClassifier()` comes in very handy when it comes to applying k nearest neighbor algorithm, it can be used for both regression & classification.<br>
`n_neighbors` specifies number of neighbors used by kNN classifier to predict a result i.e. when kNN takes input for predicting it calculates the distance(euclidean distance is default in scikit learn) of that input data point with its k(i.e. n_neighbors) nearest data points & then on basis of `algorithm` arg(which is taking majority by default) between those points it gives the prediction.
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

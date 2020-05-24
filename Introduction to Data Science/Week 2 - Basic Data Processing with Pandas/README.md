## LEARNINGS FROM ASSIGNMENT 2 BASED ON "BASIC DATA PROCESSING WITH PANDAS"
1) **BOOLEAN MASKING** is something we use very often but do not acknowledge that we actually did used a thing called "BOOLEAN MASKING". It is a basically a powerful method of filtering that we often apply when dealing with pandas dataframes.
2) **Series** is 1-D DATA STRUCTURE which consists of only index & values at those indexes i.e. just like array, just that it can have any datatype in index as well as in values. **A Series do not have multiple columns.**
3) **DataFrame** is a DATA STRUCTURE having many series(as columns) combined together, like one column represent one series.
4) **.idxmax() method** - `df['column1'].idxmax()` gives the index of the row in which value of 'column1' is maximum.
5) **.unique() method** - `df['column'].unique()` returns all unique values in a column.
6) **.groupby() method** - Whenever `df.groupby('column')` is used, index value of the dataframe becomes equal to 'column' values.
7) **.nlargest() method** - `df.groupby('column1')['column2'].nlargest(k)` gets you dataframe having the first k largest values of 'column2' with multiindex (one is 'column1' values & other is index values of the 'column2' values).
8) **.sum(level) method** - `df.groupby('column1')['column2'].nlargest(k).sum(level = 0)`, Here we specified sum(level = 0) which means sum of the first largest k values of 'column2' across level 0 (& with indices as level 0 i.e. 'column1').
9) **.max() method** - `df[['column1', 'column2', ..]].max(axis = 1)` gives the maximum values across all specified columns. Same goes for minimum i.e. this code is for **finding max or min value across various columns.**
10) `df[['column1', 'column2', ..]]` gives data of only specified columns of the dataframe.

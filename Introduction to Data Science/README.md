# LEARNINGS FROM INTRODUCTION TO DATA SCIENCE

The best thing about this course is that the instructor do not teach upto the extent at which they give the Programming assignments, so you have to search for everything to get the assignment questions correct & that is approximately same as if you're learning things by referencing documentations, stackoverflow, trying to get the code for specific work by searching query on Google. And that's the best way to learn things according to me.

I would definitely recommend this course for those who want to learn **Data Cleansing**, **Data Manipulation**, **Pandas** or **Numpy**.

This is first course of **Applied Data Science with Python Specialization** from ***Coursera***.

Link: [Introduction to Data Science  with Python](https://www.coursera.org/learn/python-data-analysis/)

Here's what things I learnt in each of the Programming Assignments:
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

## LEARNINGS FROM ASSIGNMENT 3 BASED ON "ADVANCED PYTHON PANDAS"
1) **This whole assignment was mostly based on cleaning of data. So, whenever you want to remember the things to be done while cleaning data do consider this assignment for practice or just go through it once. It really tests all necessary skills for Data Cleaning which is very important in field of Data Science.**
2) Used **Regular Expressions** to remove or replace some unwanted things by recognizing patterns in the data using regex.
3) **.rename() method** - `df.rename(index = {}, columns = {})` is used to rename rows and columns respectively in a DataFrame.
4) **pd.read_excel() function** - `pd.read_excel('file.xlsx')` is used to read an excel file as a DataFrame using pandas.
5) **pd.merge() function**
- `pd.merge(df1, df2, how = 'inner/outer', on = 'column')` returns DataFrame as a result of df1 intersection df2 & df1 union df2 respectively based on common 'column'.
- `pd.merge(df1, df2, how = 'left/right', on = 'column')` returns DataFrame with all rows of left(i.e. first argument DataFrame) & right(i.e. second argument DataFrame) respectively with all columns of both DataFrames based on common 'column'.
7) **rearrange/reorder columns** - `df = df[['column4', 'column2', 'column1', 'column3']]` can be used to rearrange columns in a DataFrame where each column should be placed at the position you want it to be.
8) **.to_frame() method** - `series.to_frame()` is used to convert a series into a DataFrame.
9) **.corr() method** - `df['col1'].corr(df['col2'], method = pearson)` is used to find correlation of 'col1' with 'col2'.
10) **.astype() method** - `df['column'].astype(int/str/float)` is used to convert 'column' of some type into some other specified datatype(wherever applicable).
11) **.map() method** - `df['newColumn'] = df['column'].map(dictionary/series)` is used to add 'newColumn' to the DataFrame by mapping(matching) keys or indexes with 'column' respectively.
12) **.agg({}) method** - `df.groupby(*cols)[*other_cols].agg({'sum':np.sum, 'mean': np.mean, ...})` is used to get statistical overview of ['other_col1', ...] using numpy functions based on grouping by ['col1', ...].
13) **pd.cut() function** - `pd.cut(df['column'], bins = k, labels = True/False)` is used to divide any 'column' into k groups of equally spaced intervals & `labels = False` converts the weird looking intervals to integers from 1 to k according to intervals.
14) **.groupby().statistical_functions()** - `df.groupby(*cols).size/sum/mean/describe...()` is used to directly get statistical overview of the DataFrame using various functions(This is only used when there is only single column in the DataFrame after grouping).
15) **.apply() method** - `df.apply(lambda x: "{:,}".format(x['column']), axis = 1)` is used to put thousand separator commas in the values of 'column', any lambda function can be used to apply on any column of the DataFrame.

## LEARNINGS FROM ASSINMENT 4 BASED ON "STATISTICAL ANALYSIS IN PYTHON"
1) **.str.replace() method** - `df['column'].str.replace(r"regex", "replacement")` is used to replace anything in all rows of a 'column', whether some specific characters or patterns by regex.
2) **.str.contains() method**
- `df[df['column'].str.contains(r"regex")]` gives the rows having some specific characters or patterns.
- `df[~df['column'].str.contains(r"regex")]` gives the rows not having the specified patterns i.e. `~` acts as a invert operator.
3) **.apply() method** - `df['column'].apply(lambda x: ...)` returns the DataFrame 'column' obtained after applying the lambda function to all the rows of the 'column'.
4) **.split() method** - `x.split(" (")[0]` gives the string first substring obtained after splitting a string by spotting ' (' in the string. `.split()` method can be applied directly on single string as well as on column of a DataFrame `df['column'].str.split()`.
5) **.strip() method**
- `x.strip()` is used to remove whitespaces from both left & right sides of x. `x.lstrip()` & `x.rstrip()` are also used to remove whitespaces from left & right respectively.
- `df['column'].str.strip(to_strip = 'pattern or characters')` is used to remove specific pattern from all the rows of the 'column' & if None then whitespaces are removed.
6) **.fillna() method** - `df.fillna(method = "ffill")` is used to ffill(forward fill - use last valid observation to fill gap) the 'NaN' values. There are other method also named bfill(backward fill - use next valid observation to fill gap). It may also be the case that you don't use any method, just fill by int, dict, series or some DataFrame. *'axis'* and *'inplace'* args are also there to use in this method.
7) **.dropna() method** - `df.dropna(how = any/all)` gives DataFrame with any row/column having NaN value dropped or any row/column having all values NaN dropped respectively. There is an arg 'subset' in which you specify the list of columns to include if you're dropping rows & vice versa. *'axis'* & *'inplace'* args are also there in this method.
8) **.append() method** - `df.append(df1)` returns a new DataFrame having 'df1' appended at the end of 'df'.
9) **.loc[] function** - `df.loc[index, 'column']` returns the row of 'column' at 'index'. It can also return multiple rows if you use boolean masking by specifying required condition in `.loc()`.
10) **.str[i1:i2]** - `df['column'].str[:4]` returns the first four characters in all rows of 'column'. This can be used if you want just some specific substring for any comparison or perform any operation on that part.

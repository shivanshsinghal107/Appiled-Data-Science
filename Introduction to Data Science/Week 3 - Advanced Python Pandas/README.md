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

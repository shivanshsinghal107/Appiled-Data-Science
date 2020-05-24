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

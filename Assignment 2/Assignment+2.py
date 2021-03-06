import pandas as pd

# Part 1

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index)
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID
# (take first 3 characters from that)

df.drop('Totals', inplace=True)


# ### Question 1
# Which country has won the most gold medals in summer games?
#
# *This function should return a single string value.*

def answer_one():
    return df['Gold'].idxmax()


# ### Question 2
# Which country had the biggest difference between their summer and
# winter gold medal counts?
#
# *This function should return a single string value.*

def answer_two():
    df['goldDiff'] = abs(df['Gold'] - df['Gold.1'])
    return df['goldDiff'].idxmax()


# ### Question 3
# Which country has the biggest difference between their summer gold medal
# counts and winter gold medal counts relative to their total gold medal count?
#
# $$\frac{Summer~Gold - Winter~Gold}{Total~Gold}$$
#
# Only include countries that have won at least 1 gold in both summer and winter.
#
# *This function should return a single string value.*

def answer_three():
    subset = df[df['Gold'] > 0]
    subset = subset[subset['Gold.1'] > 0]
    subset['relGoldDiff'] = subset['goldDiff'] / subset['Gold.2']
    return subset['relGoldDiff'].idxmax()


# ### Question 4
# Write a function that creates a Series called "Points" which is a weighted
# value where each gold medal (`Gold.2`) counts for 3 points, silver medals
# (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point.
# The function should return only the column (a Series object) which you created.
#
# *This function should return a Series named `Points` of length 146*

def answer_four():
    df['Points'] = df['Gold.2'] * 3 + df['Silver.2'] * 2 + df['Bronze.2'] * 1
    return df['Points']


# ## Part 2

# ### Question 5
# Which state has the most counties in it?
#
# *This function should return a single string value.*

def answer_five():
    census_df = pd.read_csv('census.csv')
    subset = pd.DataFrame(census_df[census_df['SUMLEV'] == 50]['STNAME'])
    subset['count'] = 0
    states = subset.groupby('STNAME').count()
    return states['count'].idxmax()


# ### Question 6
# Only looking at the three most populous counties for each state, what are the
# three most populous states (in order of highest population to lowest
# population)? Use `CENSUS2010POP`.
#
# *This function should return a list of string values.*

def answer_six():
    census_df = pd.read_csv('census.csv')
    subset = census_df[census_df['SUMLEV'] == 50][['STNAME', 'CENSUS2010POP']]
    states = pd.DataFrame()
    states['State'] = subset['STNAME'].unique()
    states['Top3Pop'] = 0
    states.set_index('State', inplace = True)

    for state in states.index:
        countyPop = subset[subset['STNAME'] == state].sort('CENSUS2010POP',
                                                           ascending = False)
        countyPop = countyPop['CENSUS2010POP'].head(3)
        if type(countyPop) == pd.Series:
            pop_sum = sum(countyPop)
        else:
            pop_sum = countyPop
        states['Top3Pop'].loc[state] = pop_sum

    output = states.sort('Top3Pop', ascending= False).head(3)
    output = list(output.index)
    return output


# ### Question 7
# Which county has had the largest absolute change in population within the
# period 2010-2015? (Hint: population values are stored in columns
# POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)
#
# e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130,
# then its largest change in the period would be |130-80| = 50.
#
# *This function should return a single string value.*

def answer_seven():
    census_df = pd.read_csv('census.csv')
    popcols = ["POPESTIMATE20"+ str(x) for x in range(10,16)]
    cols = ['CTYNAME'] + popcols
    dfp = census_df[census_df['SUMLEV'] == 50][cols]

    dfp['maxPop'] = dfp[popcols].max(axis=1)
    dfp['minPop'] = dfp[popcols].min(axis=1)
    dfp['diffPop'] = dfp['maxPop'] - dfp['minPop']

    dfp.set_index('CTYNAME', inplace = True)

    return dfp['diffPop'].idxmax()


# ### Question 8
# In this datafile, the United States is broken up into four regions using
# the "REGION" column.
#
# Create a query that finds the counties that belong to regions 1 or 2, whose
# name starts with 'Washington', and whose POPESTIMATE2015 was greater than
# their POPESTIMATE 2014.
#
# *This function should return a 5x2 DataFrame with the
# columns = ['STNAME', 'CTYNAME'] and the same index ID as
# the census_df (sorted ascending by index).*

def answer_eight():
    census_df = pd.read_csv('census.csv')
    subset = (census_df[(census_df['REGION'] < 3) & (census_df['SUMLEV'] == 50)
                        & (census_df['POPESTIMATE2015'] >
                        census_df['POPESTIMATE2014'])][['STNAME', 'CTYNAME']])
    subset = subset[[name.startswith('Washington')
                     for name in subset['CTYNAME']]]
    return subset

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
university_towns_file = 'university_towns.txt'
zillow_research_file = 'City_Zhvi_AllHomes.csv'
gdp_data_file = 'gdplev.xls'


def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    data = []
    state = ''
    with open(university_towns_file, 'r') as uni_file:
        for line in uni_file:
            line = line.replace('\n', '')
            if 'edit' in line:
                state = line.split('[')[0]
            else:
                region_name = line.split(' (')[0]
                data.append((state, region_name))
    uni_df = pd.DataFrame(data, columns=['State', 'RegionName'])
    return uni_df


def get_gdp_df():
    gdp_df = (pd.read_excel(gdp_data_file, header=4, skiprows=2)
            .filter(regex='Unnamed: [4,6]')
            .rename(columns = {'Unnamed: 4':'Quarter', 'Unnamed: 6': 'GDP'}))
    gdp_df = gdp_df[gdp_df['Quarter'].apply(lambda x: (int)(x.split('q')[0])) > 1999]
    gdp_df['Diff'] = gdp_df['GDP'].diff()
    gdp_df['Change'] = gdp_df['Diff'] < 0
    return gdp_df


def get_recession_df():
    
    gdp_df = get_gdp_df()
    rec_chk = pd.rolling_apply(gdp_df['Change'], 2, lambda x : (x[0] and x[1])).shift(-1)
    recession_start = gdp_df[rec_chk == 1.0]
    
    rest = gdp_df[gdp_df.index > recession_start.iloc[-1].name]
    rec_chk = pd.rolling_apply(rest['Change'], 2, lambda x : not(x[0] or x[1]))
    recession_end_index = rest[rec_chk == 1.0].iloc[0].name
    return recession_start.append(rest[rest.index <= recession_end_index]).reset_index()


def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    recession = get_recession_df()
    return recession.iloc[0]['Quarter']


def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format
    such as 2005q3'''
    recession = get_recession_df()
    return recession.iloc[-1]['Quarter']


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    recession = get_recession_df()
    index = recession['GDP'].idxmin()
    return recession.iloc[index]['Quarter']


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    housing_df = (pd.read_csv(zillow_research_file, 
            index_col=list(range(6)))
           .rename(columns=pd.to_datetime)
           .resample('3M', closed='left',axis=1).mean()
           .rename(columns=lambda x: '{}q{}'.format(x.year,x.quarter))
           .filter(regex='20[0-1][0-9]q[1-4]')
           .reset_index()
           .drop(['RegionID','Metro','CountyName','SizeRank'], 
                               axis=1))
    housing_df['State'] = (housing_df['State']
                                   .apply(lambda state: states.get(state)))
    housing_df.set_index(['State', 'RegionName'], inplace=True)
    return housing_df


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivalent to a
    reduced market loss).'''
    
    housing_df = convert_housing_data_to_quarters()
    rec_start = get_recession_start()
    rec_bottom = get_recession_bottom()
    housing_pr = housing_df[rec_start].div(housing_df[rec_bottom])
    uni_list = list(get_list_of_university_towns().itertuples(index=False))
    rec_housing_uni = housing_pr.loc[housing_pr.index.isin(uni_list)].dropna(how='all')
    rec_housing_non_uni = housing_pr.loc[~housing_pr.index.isin(uni_list)].dropna(how='all')

    result = ttest_ind(rec_housing_uni, rec_housing_non_uni)
    
    if result.pvalue < 0.01:
        different = True
    else:
        different = False

    mean_uni = rec_housing_uni.mean()
    mean_non_uni = rec_housing_non_uni.mean()

    if mean_non_uni < mean_uni:
        better = "non-university town"
    else:
        better = "university town"

    return (different, result.pvalue, better)

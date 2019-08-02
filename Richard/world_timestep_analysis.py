import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
from scipy import stats
import matplotlib.dates as mdates


### output formats ###

OUTPUT_TEMPLATE = (
    'P-value of linear regression results (null hypothesis: slope of the regression line is 0)\n'
    '    p-value of style "American Amber / Red Ale": {p_value1:.3g}\n'
    '    p-value of style "American Blonde Ale": {p_value2:.3g}\n'
    '    p-value of style "American IPA": {p_value3:.3g}\n'
    '    p-value of style "American Pale Ale (APA)": {p_value4:.3g}\n'
    '    p-value of style "European Pale Lager": {p_value5:.3g}\n'
)



### helper functions ###
datetime_pattern = re.compile(r'^(\S+) (\d+), (\d+)$')

def isdatetimeFormat(string): 
    # isdatetimeFormat(string) checks whether argument has good datetime format which can be converted into datetime object
    # by to_datetime(datelike_string) function. the format is "%b %d, %Y"
    match = datetime_pattern.match(string)
    if not match:
        return False
    month = match.group(1)
    date = int(match.group(2))
    year = int(match.group(3))
    if month not in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']:
        return False
    elif not( date >0 and date<32):
        return False
    elif year < 0:
        return False
    else: 
        return True

def to_datetime(datelike_string):
    ret = datelike_string[0:3] + datelike_string[8:] #datelike_string[0:3] + datelike_string[8:]
    ret = datetime.datetime.strptime(ret, "%b%Y") #(ret, "%b%Y")
    return ret

def is_year_round(string):
    ret = (string == 'Year-round') #or (string=='Rotating')
    return ret

def to_timestamp(datetime_Object):
    return datetime.datetime.timestamp(datetime_Object)


### ETL with beerinfo.csv ###

# read beerinfo.csv, only keep columns and rows required for analysis  #
beerinfo = pd.read_csv('data/new_beer.csv')
beerinfo = beerinfo[beerinfo['style'].notnull()] # style is key column for the analysis so if a row doesn't have a style, trash it
beerinfo = beerinfo[['beer_number','style','availability']] # keep only beer_number, style, and availability 
beerinfo['is_year_round'] = beerinfo['availability'].apply(is_year_round) # only keep beers brewed 'Year-around' 
beerinfo = beerinfo[beerinfo['is_year_round'] == True] 
del [beerinfo['availability'],beerinfo['is_year_round']] # availability and is_year_round is not needed anymore

# find top 5 beer styles reviewed the most #
beer_kind = beerinfo.groupby('style').count() # count how many beers are in 
beer_kind = beer_kind.sort_values(by='beer_number', ascending=False) # column 'beer_number' has same value as if it was counter for total number of beers in each style
beer_kind = beer_kind.head(5) 
print(beer_kind)

# filter the top 5 beer styles from the data #
beer_kind = beer_kind.rename(columns={'beer_number':'indicator'}) 
beerinfo = beerinfo.join(beer_kind, on='style') # non-top10 styles will have null value in column 'indicator' after the join
beerinfo = beerinfo[beerinfo['indicator'].notnull()] # filter non-top10 styles
del beerinfo['indicator'] # we dont need 'indicator' anymore


### ETL with comment.csv ###

# read data file then delete unnecessary columns  #
comment = pd.read_csv("data/new_comment.csv")
comment = comment[comment['date'].notnull()] # filtering any comment without date information
del [comment['feel'], comment['look'], comment['rdev'], comment['smell'], comment['taste'], comment['overall'], comment['username'], comment['comment']]

# making sure 'date' column has string type and it is in good format for datetime conversion #
comment['isStr'] = comment['date'].apply(isinstance, args=(str,))
comment = comment[comment['isStr']==True]
comment['format'] = comment['date'].apply(isdatetimeFormat)
comment = comment[comment['format']==True]
comment['date'] = comment['date'].apply(to_datetime) # string conversion to datetime <= this line causes warning
del [comment['isStr'], comment['format']]  # delete by-product columns

# using beerinfo dataframe, filter beers whose in top 5 styles #
comment = comment.join(beerinfo.set_index('beer_number'), on='beer_number')
comment = comment[comment['style'].notnull()]

# what are the number of reviews per year-month per style? #
counts = comment
counts = counts.rename(columns={'ba_score':'review counts'})
del counts['beer_number']
counts = counts.groupby(['style','date']).count()

# calculate mean 'ba_score' for each year-month for all beer styles #
means = comment
del means['beer_number']
means = means.groupby(['style','date']).mean()

# filter year-month with less then 30 reviews for all styles #
counts = counts.join(means , on=['style','date']) # join counts DF and means DF into one by multiindex [style,date]
del means # we dont need means anymore
counts = counts.reset_index()
counts = counts[counts['review counts'] > 29] # filter data 
# counts



### Statistical Testing ### 

# prepare for linear regression #
counts = counts.groupby(['style','date']).mean()
counts = counts.unstack(level=0)
counts = counts.reset_index()
counts['timestamp'] = counts['date'].apply(to_timestamp)

# Create mask to avoid Null values during regression #
mask1 = counts['ba_score']['American Amber / Red Ale'].notnull()
mask2 = counts['ba_score']['American Blonde Ale'].notnull()
mask3 = counts['ba_score']['American IPA'].notnull()
mask4 = counts['ba_score']['American Pale Ale (APA)'].notnull()
mask5 = counts['ba_score']['European Pale Lager'].notnull()

# Execute linear regression for ba_scores #
lin1 = stats.linregress(counts['timestamp'][mask1], counts['ba_score']['American Amber / Red Ale'][mask1])
lin2 = stats.linregress(counts['timestamp'][mask2], counts['ba_score']['American Blonde Ale'][mask2])
lin3 = stats.linregress(counts['timestamp'][mask3], counts['ba_score']['American IPA'][mask3])
lin4 = stats.linregress(counts['timestamp'][mask4], counts['ba_score']['American Pale Ale (APA)'][mask4])
lin5 = stats.linregress(counts['timestamp'][mask5], counts['ba_score']['European Pale Lager'][mask5])
prediction1 = lin1.slope * counts['timestamp'] + lin1.intercept
prediction2 = lin2.slope * counts['timestamp'] + lin2.intercept
prediction3 = lin3.slope * counts['timestamp'] + lin3.intercept
prediction4 = lin4.slope * counts['timestamp'] + lin4.intercept
prediction5 = lin5.slope * counts['timestamp'] + lin5.intercept

# create Hierarchical column 'Prediction' which contains prediction values of all styles #
predictions = {'American Amber / Red Ale':prediction1, 'American Blonde Ale':prediction2, 'American IPA':prediction3, 'American Pale Ale (APA)':prediction4, 'European Pale Lager':prediction5}
predictions = pd.DataFrame(data=predictions)
predictions.columns = pd.MultiIndex.from_product([['Prediction'], ['American Amber / Red Ale', 'American Blonde Ale', 'American IPA', 'American Pale Ale (APA)', 'European Pale Lager']]) 
counts = counts.join(predictions)
del predictions
# counts = counts.set_index('date')



### Plot and print the results ###

print(OUTPUT_TEMPLATE.format(
    p_value1=lin1.pvalue,
    p_value2=lin2.pvalue,
    p_value3=lin3.pvalue,
    p_value4=lin4.pvalue,
    p_value5=lin5.pvalue
))
plt.figure(figsize=(12,5))
plt.plot(counts['date'], counts['ba_score']['American Amber / Red Ale'], 'go', markersize=3)
plt.plot(counts['date'], counts['ba_score']['American Blonde Ale'], 'co', markersize=3)
plt.plot(counts['date'], counts['ba_score']['American IPA'], 'ro', markersize=3)
plt.plot(counts['date'], counts['ba_score']['American Pale Ale (APA)'], 'bo', markersize=3)
plt.plot(counts['date'], counts['ba_score']['European Pale Lager'], 'mo', markersize=3)
plt.plot(counts['date'], counts['Prediction']['American Amber / Red Ale'], 'g-', linewidth=3)
plt.plot(counts['date'], counts['Prediction']['American Blonde Ale'], 'c-', linewidth=3)
plt.plot(counts['date'], counts['Prediction']['American IPA'], 'r-', linewidth=3)
plt.plot(counts['date'], counts['Prediction']['American Pale Ale (APA)'], 'b-', linewidth=3)
plt.plot(counts['date'], counts['Prediction']['European Pale Lager'], 'm-', linewidth=3)
plt.legend()
plt.title('Mean beer score of 5 most reviewed beer in Beeradvocate', fontsize=25)
plt.ylabel('Beer Score [0,5]', fontsize = 18)
plt.xlabel('Year', fontsize=18)


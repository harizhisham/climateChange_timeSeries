'''
Author: harizhisham
'''

from datetime import datetime
import pandas as pd
import time

'''
Args: pandas df type containing column with timestamp format %Y-%m-%d

Output: cols of datetime features appended to original df
'''
def extract_dt(df, column):

  assert type(df) == pd.core.frame.DataFrame,\
  'Error: must pass in pandas DataFrame obj type.'
  assert type(column) == str,\
  'Error: column arg must be of str type.'

  try:
    start_time = time.time()

    df['year'] = pd.to_datetime(df[column]).dt.year.astype('int')
    df['month'] = pd.to_datetime(df[column]).dt.month.astype('int')
    df['day'] = pd.to_datetime(df[column]).dt.day.astype('int')
    df['weekday'] = pd.to_datetime(df[column]).dt.weekday.astype('int')
    df['hour'] = pd.to_datetime(df[column]).dt.hour.astype('int')

    end_time = time.time()
    total_time = end_time - start_time
    print('Total time elapsed: %.2f seconds' % total_time)

    return df

  except:
    print('Time travel does not exist.')

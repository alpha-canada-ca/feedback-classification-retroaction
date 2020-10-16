from airtable import Airtable
import pandas as pd
import pickle
from configparser import ConfigParser

#get api key from config file and get data from AirTabe
config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
base = config.get('default', 'base')
airtable = Airtable(base, 'Page feedback', api_key=key)
print('Accessed the key')
record_list = airtable.get_all()
print('Fetched the data')
#convert data to Pandas dataframe
data = pd.DataFrame([record['fields'] for record in record_list])
print('Created the dataframe')
#define serialize function
def serialize(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

#save data as a pickle
serialize(data, 'data/all_data.pickle')
print('Saved dataframe as pickle file')
print('Process complete')

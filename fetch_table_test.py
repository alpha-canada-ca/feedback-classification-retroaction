from airtable import Airtable
import pandas as pd
import pickle
from configparser import ConfigParser

#get api key from config file and get data from AirTabe
config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
base_main = config.get('default', 'base_main')
base_old = config.get('default', 'base_old')
airtable_main = Airtable(base_main, 'Page feedback', api_key=key)
airtable_old = Airtable(base_old, 'Page feedback', api_key=key)



print('Accessed the key')
record_list_main = airtable_main.get_all()
record_list_old = airtable_old.get_all()

print('Fetched the data')
#convert data to Pandas dataframe
data_main = pd.DataFrame([record['fields'] for record in record_list_main])
data_old = pd.DataFrame([record['fields'] for record in record_list_old])

data = data_main.append(data_old, ignore_index=True)


data = data[["Comment", "Date", "Status",  "What's wrong", "Lookup_tags", 'Tags confirmed', 'Yes/No', 'Lookup_page_title', 'URL_function', 'Lookup_FR_tag', "Lookup_group_EN", "Lookup_group_FR", "Lang"]]



print('Created the dataframes')
#define serialize function
def serialize(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

#save data as a pickle
serialize(data, 'data/all_data.pickle')

print('Saved dataframes as pickle files')
print('Fetching data process complete')

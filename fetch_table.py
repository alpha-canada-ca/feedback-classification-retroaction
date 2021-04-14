from airtable import Airtable
import pandas as pd
import pickle
import gzip
import pickletools
from configparser import ConfigParser

#get api key from config file and get data from AirTabe
config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
base = config.get('default', 'base')
base_health= config.get('default', 'base_health')
base_cra= config.get('default', 'base_cra')
base_travel= config.get('default', 'base_travel')
airtable_main = Airtable(base, 'Page feedback', api_key=key)
airtable_health = Airtable(base_health, 'Page feedback', api_key=key)
airtable_cra = Airtable(base_cra, 'Page feedback', api_key=key)
airtable_travel = Airtable(base_travel, 'Page feedback', api_key=key)



print('Accessed the key')
record_list_main = airtable_main.get_all()
record_list_health = airtable_health.get_all()
record_list_cra = airtable_cra.get_all()
record_list_travel = airtable_travel.get_all()

print('Fetched the data')
#convert data to Pandas dataframe
data_main = pd.DataFrame([record['fields'] for record in record_list_main])
data_health = pd.DataFrame([record['fields'] for record in record_list_health])
data_cra = pd.DataFrame([record['fields'] for record in record_list_cra])
data_travel = pd.DataFrame([record['fields'] for record in record_list_travel])

data_2 = data_main.append(data_health, ignore_index=True, sort=True)

data_1 = data_2.append(data_cra, ignore_index=True, sort=True)

data = data_1.append(data_travel, ignore_index=True, sort=True)


data = data[["Unique ID", "Comment", "Date", "Status", "Lookup_tags", 'Tags confirmed', 'Lookup_page_title', 'URL_function', 'Lookup_FR_tag', "Lookup_group_EN", "Lookup_group_FR", "Lang", "What's wrong"]]
data = data[data.Status != 'Spam']
data = data[data.Status != 'Ignore']
data = data[data.Status != 'Duplicate']

data = data.drop_duplicates(subset ="Comment")


print('Created the dataframes')

def serialize(obj, file, protocol=-1):
    with gzip.open(file, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


#save data as a pickle
serialize(data, 'data/all_data.pickle')

print('Saved dataframes as pickle files')
print('Fetching data process complete')

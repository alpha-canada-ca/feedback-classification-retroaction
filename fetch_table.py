from airtable import Airtable
import pandas as pd
import pickle
import gzip
import pickletools
from configparser import ConfigParser


def filter_data(data):
    return data[
        (data.Status != 'Spam') &
        (data.Status != 'Ignore') &
        (data.Status != 'Duplicate')
    ]


def serialize(obj, file, protocol=-1):
    with gzip.open(file, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
bases = [
    config.get('default', 'base'),
    config.get('default', 'base_health'),
    config.get('default', 'base_cra'),
    config.get('default', 'base_travel'),
    config.get('default', 'base_ircc')
]

airtables = [Airtable(base, 'Page feedback', api_key=key) for base in bases]

print('Accessed the key')
record_lists = [airtable.get_all() for airtable in airtables]

print('Fetched the data')
data = pd.concat([pd.DataFrame([record['fields'] for record in record_list])
                 for record_list in record_lists], ignore_index=True, sort=True)
data = data[["Unique ID", "Comment", "Date", "Status", "Lookup_tags", 'Tags confirmed', 'Lookup_page_title',
             'URL_function', 'Lookup_FR_tag', "Lookup_group_EN", "Lookup_group_FR", "Lang", "What's wrong"]]
data = filter_data(data)

print('Created the dataframes')

# save data as a pickle
serialize(data, 'data/all_data.pickle')

print('Saved dataframes as pickle files')
print('Fetching data process complete')

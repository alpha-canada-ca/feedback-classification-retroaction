def groupindex():

    #import libraries
    from flask import Flask
    from flask import request
    from flask import app, render_template
    import requests
    import pandas as pd
    import pickle

    #define deserialize
    def deserialize(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    #import data as pickle
    data = deserialize('data/all_data.pickle')
    lang = request.args.get('lang', 'en')

    data = data[["Lookup_group_EN", 'URL_function', 'Status', "Lookup_group_FR"]]
    data = data.dropna()
    data['Lookup_group_EN'] = [','.join(map(str, l)) for l in data['Lookup_group_EN']]
    data['Lookup_group_FR'] = [','.join(map(str, l)) for l in data['Lookup_group_FR']]
    data = data[data.Status != 'Spam']
    data = data[data.Status != 'Ignore']
    data = data[data.Status != 'Duplicate']
    data = data.drop(columns=['Status'])
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data['URL_function'] = data['URL_function'].str.replace('/content/canadasite', 'www.canada.ca')
    data['URL_function'] = data['URL_function'].str.replace('www.canada.ca', 'https://www.canada.ca')
    data['URL_function'] = data['URL_function'].str.replace('https://https://', 'https://')
    groups_fr = list(data['Lookup_group_FR'].unique())

    group_dict = {}

    group_dict = {k: g['URL_function'].tolist() for k,g in data.groupby("Lookup_group_EN")}

    group_names = data.copy()
    group_names  = group_names .drop(columns=['URL_function'])
    group_names = group_names.drop_duplicates()
    group_names = group_names.reset_index(drop=True)

    groups = list(group_dict.keys())
    pages_list = list(group_dict.values())

    groups_fr = []
    for group in groups:
      group_look = group_names[group_names["Lookup_group_EN"] == group]
      fr = list(group_look['Lookup_group_FR'])
      fr = fr[0]
      groups_fr.append(fr)


    group_list = ["bygroup?group=" + group for group in groups]
    group_list_fr = ["bygroup?group=" + group+ "&lang=fr" for group in groups]

    if lang == 'en':
        by_group_dict = {'Group':groups,'Pages':pages_list}

    if lang == 'fr':
        by_group_dict = {'Group':groups_fr,'Pages':pages_list}

    by_group_table = pd.DataFrame(by_group_dict)

    by_group_table['group_list'] = group_list

    by_group_table_fr = by_group_table.copy()
    by_group_table_fr['group_list'] = group_list_fr


    if lang == 'en':
        return render_template("group_index_en.html", by_group_table = by_group_table, zip=zip)

    if lang == 'fr':
        return render_template("group_index_fr.html", by_group_table_fr = by_group_table_fr, zip=zip)



    #split feedback by What's wrongfully

    #get most meaningful word by what's wrong

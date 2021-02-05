def bygroup():

    #import libraries
    from flask import Flask
    from flask import request
    from flask import app, render_template
    import requests
    import pandas as pd
    import pickle
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    import re
    import sys
    import warnings
    import matplotlib
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import io
    import base64
    import matplotlib.ticker as plticker
    import datetime as DT
    import gzip, pickletools

    #define deserialize
    def deserialize(file):
        with gzip.open(file, 'rb') as f:
            p = pickle.Unpickler(f)
            return p.load()

    #import data as pickle
    data = deserialize('data/all_data.pickle')


    today = DT.date.today()
    yesterday = today - DT.timedelta(days=1)
    week_ago = today - DT.timedelta(days=8)
    earliest = today - DT.timedelta(days=90)
    today = today.strftime('%F')
    yesterday = yesterday.strftime('%F')
    week_ago = week_ago.strftime('%F')
    earliest = earliest.strftime('%F')

    group = request.args.get('group', 'no_group')
    lang = request.args.get('lang', 'en')
    start_date = request.args.get('start_date', week_ago)
    end_date = request.args.get('end_date', yesterday)

    monthDict = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06',
                'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}



    if lang == 'en':
        tag_columns = ['Date', 'Comment']


    if lang == 'fr':
        tag_columns = ['Date', 'Commentaire']


    if group == 'no_group':

        return render_template("no_page.html")

    else:

        data['URL_function'] = data['URL_function'].str.replace('/content/canadasite', 'www.canada.ca')
        data['URL_function'] = data['URL_function'].str.replace('www.canada.ca', 'https://www.canada.ca')
        data['URL_function'] = data['URL_function'].str.replace('https://https://', 'https://')
        data['Lookup_group_EN'].fillna("None", inplace=True)
        data['Lookup_group_EN'] = [','.join(map(str, l)) for l in data['Lookup_group_EN']]
        group_data = data.loc[data['Lookup_group_EN'] == group]
        group_data = group_data.reset_index(drop=True)
        group_data = group_data[group_data['Date'] >= earliest]

        if group_data.empty:

            return render_template("empty.html")

        else:

            group_data = group_data[["Comment", "Date", "Status",  "What's wrong", "Lookup_tags", 'Tags confirmed', 'Lookup_page_title', 'URL_function', 'Lookup_FR_tag', "Lookup_group_EN", "Lookup_group_FR" ]]
            group_data = group_data[group_data.Status != 'Spam']
            group_data = group_data[group_data.Status != 'Ignore']
            group_data = group_data[group_data.Status != 'Duplicate']
            group_data = group_data.reset_index(drop=True)

            group_data['Lookup_page_title'] = [','.join(map(str, l)) for l in group_data['Lookup_page_title']]
            group_data['Lookup_group_FR'] = [','.join(map(str, l)) for l in group_data['Lookup_group_FR']]

            urls = group_data[['URL_function', 'Lookup_page_title']]
            urls = urls.drop_duplicates()

            if lang == "en":
                group_name = group_data['Lookup_group_EN'][0]

            if lang == "fr":
                group_name = group_data['Lookup_group_FR'][0]


            group_data["What's wrong"].fillna(False, inplace=True)
            group_data["Tags confirmed"].fillna(False, inplace=True)
            if lang == 'en':
                group_data['Lookup_tags'] = group_data['Lookup_tags'].apply(lambda d: d if isinstance(d, list) else ['Untagged'])
            if lang == 'fr':
                group_data['Lookup_tags'] = group_data['Lookup_tags'].apply(lambda d: d if isinstance(d, list) else ['Non-étiquettés'])


            if lang == 'en':
                group_data['Lookup_FR_tag'] = group_data['Lookup_FR_tag'].apply(lambda d: d if isinstance(d, list) else ['Untagged'])
            if lang == 'fr':
                group_data['Lookup_FR_tag'] = group_data['Lookup_FR_tag'].apply(lambda d: d if isinstance(d, list) else ['Non-étiquettés'])
            all_data = group_data.copy()


            #limit page_data to period

            group_data = group_data[group_data['Date'] <= end_date]
            group_data = group_data[group_data['Date'] >= start_date]


            # only keep commments

            all_data = all_data.dropna()

            if all_data.empty:

                if lang == 'en':
                    return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, lang = lang, zip = zip, group = group, urls=urls)

                if lang == 'fr':
                    return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, lang = lang, zip = zip, group = group, urls=urls)

            else:

                if lang == 'en':
                    all_data['tags'] = [','.join(map(str, l)) for l in all_data['Lookup_tags']]
                if lang == 'fr':
                    all_data['tags'] = [','.join(map(str, l)) for l in all_data['Lookup_FR_tag']]

                #remove the Lookup_tags column (it's not needed anymore)
                all_data = all_data.drop(columns=['Lookup_tags'])
                all_data = all_data.drop(columns=['Lookup_FR_tag'])
                all_tags = all_data["tags"].str.split(",", n = 3, expand = True)
                all_data = all_data.join(all_tags)
                all_data = all_data.drop(columns=['tags'])

                tag_count = all_tags.apply(pd.Series.value_counts)
                tag_count = tag_count.fillna(0)
                tag_count = tag_count.astype(int)
                if 2 in tag_count.columns:
                    tag_count = tag_count[0] + tag_count[1] + tag_count[2]
                elif 1 in tag_count.columns:
                    tag_count = tag_count[0] + tag_count[1]
                else:
                    tag_count = tag_count[0]
                tag_count = tag_count.sort_values(ascending = False)
                by_tag = tag_count.to_frame()
                by_tag = by_tag.sort_index(axis=0, level=None, ascending=True)
                by_tag.columns = ['Feedback count']

                by_tag = by_tag.sort_values(by = 'Feedback count', ascending=False)
                unique_tags = list(by_tag.index)


                tag_dico = {}

                tag_dico_columns = ['Date', 'Comment', 'URL_function']

                for tag in unique_tags:
                  tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                for tag, topic_df in all_data.groupby(0):
                  tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]


                if 1 in all_data.columns:
                    for tag, topic_dfn in all_data.groupby(1):
                        if tag_dico[tag].empty:
                            tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                        else:
                            tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])

                if 2 in all_data.columns:
                    for tag, topic_df in all_data.groupby(2):
                        if tag_dico[tag].empty:
                            tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                        else:
                            tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])


                for tag in tag_dico:
                    tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                tag_dates = {}

                for tag in tag_dico:
                  tag_dates[tag] = tag_dico[tag]['Date'].value_counts()



                date_range = all_data['Date']
                date_range = date_range.sort_values()
                date_range = date_range.reset_index(drop=True)

                for tag in tag_dates:
                  idx = pd.date_range(date_range.iloc[0], date_range.iloc[-1])
                  tag_dates[tag].index = pd.DatetimeIndex(tag_dates[tag].index)
                  tag_dates[tag] = tag_dates[tag].reindex(idx, fill_value=0)


                tag_plots = {}
                for tag in tag_dates:
                  tag_dates[tag]= tag_dates[tag].to_frame()
                  tag_dates[tag].reset_index(level=0, inplace=True)
                  tag_dates[tag].columns = ['Date', 'Count']
                  tag_dates[tag]['Rolling mean'] = tag_dates[tag].iloc[:,1].rolling(window=7).mean()
                  dates = list(tag_dates[tag]['Date'])
                  daily_values =  list(tag_dates[tag]['Count'])
                  weekly_values = list(tag_dates[tag]['Rolling mean'])
                  column = tag_dates[tag]['Count']
                  high_y = column.max()
                  max_y = high_y + 5
                  start_plot = start_date
                  end_plot = end_date
                  all_start = dates[0]
                  all_end = dates[-1]
                  img = io.BytesIO()
                  x = dates
                  y1 = daily_values
                  y2 = weekly_values
                  fig, ax = plt.subplots()
                  if lang == 'en':
                      ax.bar(x, y1, color=(0.2, 0.4, 0.6, 0.6), linewidth=0.5, label='Daily value')
                      ax.plot(x, y2, color='black', linewidth=3.0, label='Weekly rolling mean')
                      plt.title(tag + '\n' + 'Number of commments per day')
 
                  if lang == 'fr':
                      ax.bar(x, y1, color=(0.2, 0.4, 0.6, 0.6), linewidth=0.5, label='Valeur quotidienne')
                      ax.plot(x, y2, color='black', linewidth=3.0, label='Moyenne mobile sur 7 jours')
                      plt.title(tag + '\n' + 'Nombre de commentaires par jour   ')

                  plt.axvspan(start_plot, end_plot, color='blue', alpha=0.3)
                  ax.legend()
                  plt.ylim(0, max_y)
                  loc = plticker.MultipleLocator(base=7.0)
                  plt.gcf().subplots_adjust(bottom=0.2)
                  fig.autofmt_xdate()

                  ax.xaxis.set_major_locator(loc)
                  fig.savefig(img, format='png')
                  plt.close()
                  img.seek(0)
                  tag_plots[tag] = base64.b64encode(img.getvalue()).decode()
                  plt.clf()

                plots = list(tag_plots.values())


                #look at comments for period

                group_data = group_data.drop(columns=['Status'])
                group_data["What's wrong"].fillna(False, inplace=True)
                group_data["Tags confirmed"].fillna(False, inplace=True)

                group_data = group_data.dropna()

                unconfirmed = group_data.loc[group_data['Tags confirmed'] == False]

                #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
                confirmed = group_data.loc[group_data['Tags confirmed'] == True]


                if confirmed.empty and unconfirmed.empty :


                    if lang == 'en':
                        return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, lang = lang, zip = zip, group = group, urls=urls)

                    if lang == 'fr':
                        return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, lang = lang, zip = zip, group = group, urls=urls)


                elif confirmed.empty:

                    unconfirmed = unconfirmed.reset_index(drop=True)
                    if lang == "en":
                        unconfirmed['tags'] = [','.join(map(str, l)) for l in unconfirmed['Lookup_tags']]
                    if lang == "fr":
                        unconfirmed['tags'] = [','.join(map(str, l)) for l in unconfirmed['Lookup_FR_tag']]

                    unconfirmed = unconfirmed.drop(columns=['Lookup_tags'])
                    unconfirmed = unconfirmed.drop(columns=['Lookup_FR_tag'])
                    unconfirmed = unconfirmed.drop(columns=['Tags confirmed'])
                    unconfirmed = unconfirmed.reset_index(drop=True)
                    unconfirmed_tags = unconfirmed["tags"].str.split(",", n = 3, expand = True)
                    unconfirmed = unconfirmed.join(unconfirmed_tags)
                    unconfirmed = unconfirmed.drop(columns=['tags'])

                    unconfirmed_tag_count = unconfirmed_tags.apply(pd.Series.value_counts)
                    unconfirmed_tag_count = unconfirmed_tag_count.fillna(0)
                    unconfirmed_tag_count = unconfirmed_tag_count.astype(int)
                    if 2 in unconfirmed_tag_count.columns:
                        unconfirmed_tag_count = unconfirmed_tag_count[0] + unconfirmed_tag_count[1] + unconfirmed_tag_count[2]
                    elif 1 in unconfirmed_tag_count.columns:
                        unconfirmed_tag_count = unconfirmed_tag_count[0] + unconfirmed_tag_count[1]
                    else:
                        unconfirmed_tag_count = unconfirmed_tag_count[0]
                    unconfirmed_tag_count = unconfirmed_tag_count.sort_values(ascending = False)
                    unconfirmed_by_tag = unconfirmed_tag_count.to_frame()
                    unconfirmed_by_tag = unconfirmed_by_tag.sort_index(axis=0, level=None, ascending=True)
                    unconfirmed_by_tag.columns = ['Feedback count']

                    unconfirmed_by_tag = unconfirmed_by_tag.sort_values(by = 'Feedback count', ascending=False)
                    unconfirmed_unique_tags = list(unconfirmed_by_tag.index)
                    unconfirmed_tag_dico = {}

                    tag_dico_columns = ['Date', 'Comment', 'URL_function']

                    for tag in unconfirmed_unique_tags:
                      unconfirmed_tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                    for tag, topic_df in unconfirmed.groupby(0):
                      unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]


                    if 1 in unconfirmed.columns:
                        for tag, topic_df in unconfirmed.groupby(1):
                            if unconfirmed_tag_dico[tag].empty:
                                unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                            else:
                                unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])

                    if 2 in unconfirmed.columns:
                        for tag, topic_df in unconfirmed.groupby(2):
                            if unconfirmed_tag_dico[tag].empty:
                                unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                            else:
                                unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])


                    for tag in unconfirmed_tag_dico:
                        unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].sort_values(by = 'Date', ascending=False)


                    unconfirmed_dict = { key: unconfirmed_tag_dico[key] for key in unconfirmed_unique_tags }

                    tag_columns = ['Date', 'Comment', 'URL_function']

                    unconfirmed_tag_plots = { tag: tag_plots[tag] for tag in unconfirmed_unique_tags }
                    unconfirmed_plots = list(unconfirmed_tag_plots.values())

                    if lang == 'en':

                        return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, list = list, tag_columns = tag_columns, lang = lang, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, zip=zip, group = group, urls=urls)

                    if lang == 'fr':
                        return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, list = list, tag_columns = tag_columns, lang = lang, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, zip=zip, group = group, urls=urls)

                    #remove the Lookup_page_title column (it's not needed anymore)

                elif unconfirmed.empty:


                    confirmed = confirmed.drop(columns=['Tags confirmed'])

                    #resets the index for each row - needed for further processing
                    confirmed= confirmed.reset_index(drop=True)

                    #split dataframe for French comments - same comments as above for each line

                    #get data for specific page

                    #split tags and expand
                    if lang == 'en':
                        confirmed['tags'] = [','.join(map(str, l)) for l in confirmed['Lookup_tags']]
                    if lang == 'fr':
                        confirmed['tags'] = [','.join(map(str, l)) for l in confirmed['Lookup_FR_tag']]

                    tags = confirmed["tags"].str.split(",", n = 3, expand = True)
                    confirmed = confirmed.join(tags)
                    confirmed = confirmed.drop(columns=['tags'])

                    confirmed = confirmed.reset_index(drop=True)


                    #count the number for each tag
                    tag_count = tags.apply(pd.Series.value_counts)
                    tag_count = tag_count.fillna(0)
                    tag_count = tag_count.astype(int)
                    if 2 in tag_count.columns:
                        tag_count = tag_count[0] + tag_count[1] + tag_count[2]
                    elif 1 in tag_count.columns:
                        tag_count = tag_count[0] + tag_count[1]
                    else:
                        tag_count = tag_count[0]
                    tag_count = tag_count.sort_values(ascending = False)
                    by_tag = tag_count.to_frame()
                    by_tag = by_tag.sort_index(axis=0, level=None, ascending=True)
                    by_tag.columns = ['Feedback count']

                    by_tag = by_tag.sort_values(by = 'Feedback count', ascending=False)
                    unique_tags = list(by_tag.index)

                    #split feedback by tag

                    tag_dico = {}

                    tag_dico_columns = ['Date', 'Comment', 'URL_function']

                    for tag in unique_tags:
                      tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                    for tag, topic_df in confirmed.groupby(0):
                      tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]


                    if 1 in confirmed.columns:
                        for tag, topic_df in confirmed.groupby(1):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])

                    if 2 in confirmed.columns:
                        for tag, topic_df in confirmed.groupby(2):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])


                    for tag in tag_dico:
                        tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                    over_tags = by_tag[(by_tag > 3).any(1)]
                    under_tags = by_tag[(by_tag <= 3).any(1)]

                    over_unique_tags = list(over_tags.index)
                    under_unique_tags = list(under_tags.index)

                    over_dict = { key: tag_dico[key] for key in over_unique_tags }
                    under_dict = { key: tag_dico[key] for key in under_unique_tags }

                    tag_columns = ['Date', 'Comment', 'URL_function']


                    over_tag_plots = { tag: tag_plots[tag] for tag in over_unique_tags }
                    over_plots = list(over_tag_plots.values())


                    under_tag_plots = { tag: tag_plots[tag] for tag in under_unique_tags }
                    under_plots = list(under_tag_plots.values())

                    #split feedback by tag


                    if lang == 'en':

                        return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, zip = zip, list = list, tag_columns = tag_columns, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, lang = lang, group = group, urls=urls)

                    if lang == 'fr':
                        return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, zip = zip, list = list, tag_columns = tag_columns, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, lang = lang, group = group, urls=urls)


                else:

                        confirmed = confirmed.drop(columns=['Tags confirmed'])

                        #resets the index for each row - needed for further processing
                        confirmed= confirmed.reset_index(drop=True)

                        #split dataframe for French comments - same comments as above for each line

                        #get data for specific page

                        #split tags and expand
                        if lang == 'en':
                            confirmed['tags'] = [','.join(map(str, l)) for l in confirmed['Lookup_tags']]
                        if lang == 'fr':
                            confirmed['tags'] = [','.join(map(str, l)) for l in confirmed['Lookup_FR_tag']]

                        tags = confirmed["tags"].str.split(",", n = 3, expand = True)
                        confirmed = confirmed.join(tags)
                        confirmed = confirmed.drop(columns=['tags'])

                        confirmed = confirmed.reset_index(drop=True)


                        #count the number for each tag
                        tag_count = tags.apply(pd.Series.value_counts)
                        tag_count = tag_count.fillna(0)
                        tag_count = tag_count.astype(int)
                        if 2 in tag_count.columns:
                            tag_count = tag_count[0] + tag_count[1] + tag_count[2]
                        elif 1 in tag_count.columns:
                            tag_count = tag_count[0] + tag_count[1]
                        else:
                            tag_count = tag_count[0]
                        tag_count = tag_count.sort_values(ascending = False)
                        by_tag = tag_count.to_frame()
                        by_tag = by_tag.sort_index(axis=0, level=None, ascending=True)
                        by_tag.columns = ['Feedback count']

                        by_tag = by_tag.sort_values(by = 'Feedback count', ascending=False)
                        unique_tags = list(by_tag.index)

                        #split feedback by tag

                        tag_dico = {}

                        tag_dico_columns = ['Date', 'Comment', 'URL_function']

                        for tag in unique_tags:
                          tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df in confirmed.groupby(0):
                          tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]


                        if 1 in confirmed.columns:
                            for tag, topic_df in confirmed.groupby(1):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                                else:
                                    tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])

                        if 2 in confirmed.columns:
                            for tag, topic_df in confirmed.groupby(2):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                                else:
                                    tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])


                        for tag in tag_dico:
                            tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        over_tags = by_tag[(by_tag > 3).any(1)]
                        under_tags = by_tag[(by_tag <= 3).any(1)]

                        over_unique_tags = list(over_tags.index)
                        under_unique_tags = list(under_tags.index)

                        over_dict = { key: tag_dico[key] for key in over_unique_tags }
                        under_dict = { key: tag_dico[key] for key in under_unique_tags }

                        tag_columns = ['Date', 'Comment', 'URL_function']


                        over_tag_plots = { tag: tag_plots[tag] for tag in over_unique_tags }
                        over_plots = list(over_tag_plots.values())


                        under_tag_plots = { tag: tag_plots[tag] for tag in under_unique_tags }
                        under_plots = list(under_tag_plots.values())


                        unconfirmed = unconfirmed.reset_index(drop=True)
                        if lang == "en":
                            unconfirmed['tags'] = [','.join(map(str, l)) for l in unconfirmed['Lookup_tags']]
                        if lang == "fr":
                            unconfirmed['tags'] = [','.join(map(str, l)) for l in unconfirmed['Lookup_FR_tag']]

                        unconfirmed = unconfirmed.drop(columns=['Lookup_tags'])
                        unconfirmed = unconfirmed.drop(columns=['Lookup_FR_tag'])
                        unconfirmed = unconfirmed.drop(columns=['Tags confirmed'])
                        unconfirmed = unconfirmed.reset_index(drop=True)
                        unconfirmed_tags = unconfirmed["tags"].str.split(",", n = 3, expand = True)
                        unconfirmed = unconfirmed.join(unconfirmed_tags)
                        unconfirmed = unconfirmed.drop(columns=['tags'])

                        unconfirmed_tag_count = unconfirmed_tags.apply(pd.Series.value_counts)
                        unconfirmed_tag_count = unconfirmed_tag_count.fillna(0)
                        unconfirmed_tag_count = unconfirmed_tag_count.astype(int)
                        if 2 in unconfirmed_tag_count.columns:
                            unconfirmed_tag_count = unconfirmed_tag_count[0] + unconfirmed_tag_count[1] + unconfirmed_tag_count[2]
                        elif 1 in unconfirmed_tag_count.columns:
                            unconfirmed_tag_count = unconfirmed_tag_count[0] + unconfirmed_tag_count[1]
                        else:
                            unconfirmed_tag_count = unconfirmed_tag_count[0]
                        unconfirmed_tag_count = unconfirmed_tag_count.sort_values(ascending = False)
                        unconfirmed_by_tag = unconfirmed_tag_count.to_frame()
                        unconfirmed_by_tag = unconfirmed_by_tag.sort_index(axis=0, level=None, ascending=True)
                        unconfirmed_by_tag.columns = ['Feedback count']

                        unconfirmed_by_tag = unconfirmed_by_tag.sort_values(by = 'Feedback count', ascending=False)
                        unconfirmed_unique_tags = list(unconfirmed_by_tag.index)
                        unconfirmed_tag_dico = {}

                        tag_dico_columns = ['Date', 'Comment', 'URL_function']

                        for tag in unconfirmed_unique_tags:
                          unconfirmed_tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df in unconfirmed.groupby(0):
                          unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]


                        if 1 in unconfirmed.columns:
                            for tag, topic_df in unconfirmed.groupby(1):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])

                        if 2 in unconfirmed.columns:
                            for tag, topic_df in unconfirmed.groupby(2):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL_function']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL_function']])


                        for tag in unconfirmed_tag_dico:
                            unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        unconfirmed_dict = { key: unconfirmed_tag_dico[key] for key in unconfirmed_unique_tags }

                        tag_columns = ['Date', 'Comment', 'URL_function']

                        unconfirmed_tag_plots = { tag: tag_plots[tag] for tag in unconfirmed_unique_tags }
                        unconfirmed_plots = list(unconfirmed_tag_plots.values())

                        if lang == 'en':

                            return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, list = list, tag_columns = tag_columns, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, lang = lang, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, zip = zip, group = group, urls = urls)

                        if lang == 'fr':
                            return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, list = list, tag_columns = tag_columns, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, lang = lang, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, zip = zip, group = group, urls = urls)

def bygroup():

    #import libraries
    from flask import Flask
    from flask import request
    from flask import app, render_template
    import requests
    import pandas as pd
    import pickle
    from nltk.corpus import stopwords
    import nltk
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

    #define deserialize
    def deserialize(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    #import data as pickle
    data = deserialize('data/all_data.pickle')
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=7)
    today = today.strftime('%F')
    week_ago = week_ago.strftime('%F')

    group = request.args.get('group', 'no_group')
    lang = request.args.get('lang', 'en')
    start_date = request.args.get('start_date', week_ago)
    end_date = request.args.get('end_date', today)


    if lang == 'en':
        tag_columns = ['Date', 'Comment']
        chart_columns = ['Date', 'Yes', 'No', 'Daily percentage (%)', 'Weekly rolling mean (%)']

    if lang == 'fr':
        tag_columns = ['Date', 'Commentaire']
        reason_column_names = ['Nombre de rétroactions', 'Raison', 'Mots significatifs']
        chart_columns = ['Date', 'Oui', 'Non', 'Pourcentage quotidien (%)', 'Moyenne mobile sur 7 jours (%)']


    if group == 'no_group':

        return render_template("no_page.html")

    else:

        data['URL'] = data['URL'].str.replace('/content/canadasite', 'www.canada.ca')
        data['URL'] = data['URL'].str.replace('www.canada.ca', 'https://www.canada.ca')
        data['URL'] = data['URL'].str.replace('https://https://', 'https://')
        group_data = data.loc[data['Main section'] == group]
        group_data = group_data.reset_index(drop=True)

        if group_data.empty:

            return render_template("empty.html")

        else:

            group_data = group_data[["Comment", "Date", "Status",  "What's wrong", "Lookup_tags", 'Tags confirmed', 'Yes/No', 'Lookup_page_title', 'URL', 'Lookup_FR_tag' ]]
            group_data = group_data[group_data.Status != 'Spam']
            group_data = group_data[group_data.Status != 'Ignore']
            group_data = group_data[group_data.Status != 'Duplicate']
            group_data = group_data.reset_index(drop=True)

            group_data['Lookup_page_title'] = [','.join(map(str, l)) for l in group_data['Lookup_page_title']]

            group_name = group

            #yes_no for all period

            yes_no = group_data[["Date", 'Yes/No']]
            yes_no = yes_no.dropna()
            yes_no = yes_no.sort_values(by = 'Date', ascending=False)
            yes_no = yes_no.reset_index(drop=True)

            by_date = {}
            for date in yes_no['Date']:
              by_date[date] = yes_no.loc[yes_no['Date'] == date]

            for date in by_date:
              by_date[date] =  by_date[date]['Yes/No'].value_counts()

            for date in by_date:
              if 'No' not in by_date[date]:
                by_date[date]['No'] = 0

            for date in by_date:
              if 'Yes' not in by_date[date]:
                by_date[date]['Yes'] = 0

            for date in by_date:
              by_date[date] = [by_date[date]['Yes'], by_date[date]['No'], (by_date[date]['Yes']/(by_date[date]['Yes'] + by_date[date]['No'])) * 100]


            df_yes = pd.DataFrame(list(by_date.values()),columns = ['Yes', 'No', 'Percentage'])
            df_yes['Date'] = list(by_date.keys())
            df_yes = df_yes[['Date', 'Yes', 'No', 'Percentage']]

            df_yes = df_yes.sort_values(by = 'Date')
            df_yes['Rolling mean'] = df_yes.iloc[:,3].rolling(window=7).mean()
            dates = list(df_yes['Date'])
            dates_r = dates[::-1]
            chart_yes = list(df_yes['Yes'])
            chart_yes = chart_yes[::-1]
            chart_no = list(df_yes['No'])
            chart_no = chart_no[::-1]
            daily_values = list(df_yes['Percentage'])
            weekly_values = list(df_yes['Rolling mean'])
            daily_perc = ["%.2f" % number for number in daily_values]
            weekly_perc = ["%.2f" % number for number in weekly_values]
            daily_perc_r = daily_perc[::-1]
            weekly_perc_r = weekly_perc[::-1]


            start_plot = start_date
            end_plot = end_date

            if start_plot < dates[0]:
                start_plot = dates[0]

            if end_plot > dates[-1]:
                end_plot = dates[-1]

            all_start = dates[0]
            all_end = dates[-1]

            img = io.BytesIO()
            x = dates
            y1 = daily_values
            y2 = weekly_values
            fig, ax = plt.subplots()
            plt.xticks(rotation=45)
            plt.ylim(0, 100)
            if lang == 'en':
                ax.plot(x, y1, linewidth=0.5, label='Daily value')
                ax.plot(x, y2, linewidth=3.0, label='Weekly rolling mean')
                plt.title('Percentage of people who said they found their answer')

            if lang == 'fr':
                ax.plot(x, y1, linewidth=0.5, label='Valeur quotidienne')
                ax.plot(x, y2, linewidth=3.0, label='Moyenne mobile sur 7 jours')
                plt.title('Pourcentage de gens qui disent avoir trouver leur réponse')

            plt.axvspan(start_plot, end_plot, color='blue', alpha=0.3)
            plt.legend()
            loc = plticker.MultipleLocator(base=7.0)
            plt.gcf().subplots_adjust(bottom=0.2)
            ax.xaxis.set_major_locator(loc)
            fig.savefig(img, format='png')
            plt.close()
            img.seek(0)

            plot_url = base64.b64encode(img.getvalue()).decode()

            if yes_no.empty:
                score = 'unavailable'
                yes = 'unavailable'
                no = 'unavailable'
            else:
                total = yes_no['Yes/No'].value_counts()
                if 'Yes' in total:
                  yes = total['Yes']
                else:
                  yes= 0

                if 'No' in total:
                  no= total['No']
                else:
                  no = 0
                score = (yes/ ( yes +  no)) * 100
                score = format(score, '.2f')



            group_data["What's wrong"].fillna(False, inplace=True)
            group_data["Tags confirmed"].fillna(False, inplace=True)



            all_data = group_data.copy()


            #limit page_data to period

            group_data = group_data[group_data['Date'] <= end_date]
            group_data = group_data[group_data['Date'] >= start_date]


            yes_no_period = group_data[["Date", 'Yes/No']]
            yes_no_period = yes_no_period.dropna()
            yes_no_period = yes_no_period.sort_values(by = 'Date', ascending=False)
            yes_no_period = yes_no_period.reset_index(drop=True)

            if yes_no_period.empty:
                score_period = 'unavailable'
                yes_period = 'unavailable'
                no_period = 'unavailable'
                delta = 'unavailable'
            else:
                total_period = yes_no_period['Yes/No'].value_counts()
                if 'Yes' in total_period:
                  yes_period = total_period['Yes']
                else:
                  yes_period = 0

                if 'No' in total_period:
                  no_period = total_period['No']
                else:
                  no_period = 0

                score_period = (yes_period / ( yes_period +  no_period)) * 100
                score_period = format(score_period, '.2f')
                if score_period > score:
                    delta = '+' + format(float(score_period)-float(score), '.2f')
                elif score_period < score:
                    delta =  format(float(score_period)-float(score), '.2f')
                else:
                    delta = 'no change'


            # only keep commments

            all_data = all_data.drop(columns=['Yes/No'])
            all_data = all_data.dropna()

            if all_data.empty:

                if lang == 'en':
                    return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group = group)

                if lang == 'fr':
                    return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group = group)

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

                tag_dico_columns = ['Date', 'Comment', 'URL']

                for tag in unique_tags:
                  tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                for tag, topic_df in all_data.groupby(0):
                  tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]


                if 1 in all_data.columns:
                    for tag, topic_dfn in all_data.groupby(1):
                        if tag_dico[tag].empty:
                            tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]
                        else:
                            tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL']])

                if 2 in all_data.columns:
                    for tag, topic_df in all_data.groupby(2):
                        if tag_dico[tag].empty:
                            tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]
                        else:
                            tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL']])


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
                  plt.legend()
                  plt.ylim(0, max_y)
                  loc = plticker.MultipleLocator(base=7.0)
                  plt.gcf().subplots_adjust(bottom=0.2)
                  fig.autofmt_xdate()

                  ax.xaxis.set_major_locator(loc)
                  fig.savefig(img, format='png')
                  plt.close()
                  img.seek(0)
                  tag_plots[tag] = base64.b64encode(img.getvalue()).decode()

                plots = list(tag_plots.values())



                #look at comments for period

                group_data = group_data.drop(columns=['Status'])
                group_data = group_data.drop(columns=['Yes/No'])
                group_data["What's wrong"].fillna(False, inplace=True)
                group_data["Tags confirmed"].fillna(False, inplace=True)

                group_data = group_data.dropna()

                unconfirmed = group_data.loc[group_data['Tags confirmed'] == False]

                #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
                group_data = group_data.loc[group_data['Tags confirmed'] == True]

                if group_data.empty:

                    if lang == 'en':
                        return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group = group)

                    if lang == 'fr':
                        return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group = group)

                else:





                    #remove the Lookup_page_title column (it's not needed anymore)
                    group_data = group_data.drop(columns=['Tags confirmed'])

                    #resets the index for each row - needed for further processing
                    group_data = group_data.reset_index(drop=True)

                    #split dataframe for French comments - same comments as above for each line

                    #get data for specific page

                    #split tags and expand
                    if lang == 'en':
                        group_data['tags'] = [','.join(map(str, l)) for l in group_data['Lookup_tags']]
                    if lang == 'fr':
                        group_data['tags'] = [','.join(map(str, l)) for l in group_data['Lookup_FR_tag']]

                    tags = group_data["tags"].str.split(",", n = 3, expand = True)
                    group_data = group_data.join(tags)
                    group_data = group_data.drop(columns=['tags'])

                    group_data = group_data.reset_index(drop=True)


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

                    tag_dico_columns = ['Date', 'Comment', 'URL']

                    for tag in unique_tags:
                      tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                    for tag, topic_df in group_data.groupby(0):
                      tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]


                    if 1 in group_data.columns:
                        for tag, topic_df in group_data.groupby(1):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL']])

                    if 2 in group_data.columns:
                        for tag, topic_df in group_data.groupby(2):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL']])


                    for tag in tag_dico:
                        tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                    over_tags = by_tag[(by_tag > 3).any(1)]
                    under_tags = by_tag[(by_tag <= 3).any(1)]

                    over_unique_tags = list(over_tags.index)
                    under_unique_tags = list(under_tags.index)

                    over_dict = { key: tag_dico[key] for key in over_unique_tags }
                    under_dict = { key: tag_dico[key] for key in under_unique_tags }

                    tag_columns = ['Date', 'Comment', 'URL']


                    over_tag_plots = { tag: tag_plots[tag] for tag in over_unique_tags }
                    over_plots = list(over_tag_plots.values())


                    under_tag_plots = { tag: tag_plots[tag] for tag in under_unique_tags }
                    under_plots = list(under_tag_plots.values())

                    #split feedback by tag

                    if unconfirmed.empty:

                        if lang == 'en':

                            return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, zip = zip, page = page, list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, group = group)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score,  list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group = group)

                    else:

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

                        tag_dico_columns = ['Date', 'Comment', 'URL']

                        for tag in unconfirmed_unique_tags:
                          unconfirmed_tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df in unconfirmed.groupby(0):
                          unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]


                        if 1 in unconfirmed.columns:
                            for tag, topic_df in unconfirmed.groupby(1):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL']])

                        if 2 in unconfirmed.columns:
                            for tag, topic_df in unconfirmed.groupby(2):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df[['Date', 'Comment', 'URL']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df[['Date', 'Comment', 'URL']])


                        for tag in unconfirmed_tag_dico:
                            unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        unconfirmed_dict = { key: unconfirmed_tag_dico[key] for key in unconfirmed_unique_tags }

                        tag_columns = ['Date', 'Comment', 'URL']

                        unconfirmed_tag_plots = { tag: tag_plots[tag] for tag in unconfirmed_unique_tags }
                        unconfirmed_plots = list(unconfirmed_tag_plots.values())

                        if lang == 'en':

                            return render_template("by_group_en.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags,  list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, zip=zip, group = group)

                        if lang == 'fr':
                            return render_template("by_group_fr.html", group_name = group_name, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, zip=zip, group = group)

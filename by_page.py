def bypage():

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
    from wordcloud import WordCloud

    #define deserialize
    def deserialize(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    #import data as pickle
    data = deserialize('data/all_data.pickle')
    yes_no_db = deserialize('data/yes_no_db.pickle')
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=7)
    earliest = today - DT.timedelta(days=90)
    today = today.strftime('%F')
    week_ago = week_ago.strftime('%F')
    earliest = earliest.strftime('%F')

    page = request.args.get('page', 'no_page')
    lang = request.args.get('lang', 'en')
    start_date = request.args.get('start_date', week_ago)
    end_date = request.args.get('end_date', today)

    monthDict = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 
                'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}


    if lang == 'en':
        tag_columns = ['Date', 'Comment']
        reason_column_names = ['Feedback count', 'Reason']
        word_column_names = ['Count', 'Word']
        chart_columns = ['Date', 'Yes', 'No', 'Daily ratio', 'Weekly rolling mean']

    if lang == 'fr':
        tag_columns = ['Date', 'Commentaire']
        reason_column_names = ['Nombre de rétroactions', 'Raison']
        word_column_names = ['Nombre', 'Mots']
        chart_columns = ['Date', 'Oui', 'Non', 'Proportion quotidienne', 'Moyenne mobile sur 7 jours']


    if page == 'no_page':

        return render_template("no_page.html")

    else:

        data['URL_function'] = data['URL_function'].str.replace('/content/canadasite', 'www.canada.ca')
        data['URL_function'] = data['URL_function'].str.replace('www.canada.ca', 'https://www.canada.ca')
        data['URL_function'] = data['URL_function'].str.replace('https://https://', 'https://')
        # gets page data for current page
        page_data = data.loc[data['URL_function'] == page]
        # Reset indexes (0,1,2,3.. etc)
        page_data = page_data.reset_index(drop=True)
        # grab data from 90 days ago onwards
        page_data = page_data[page_data['Date'] >= earliest]

        if page_data.empty:

            return render_template("empty.html")

        else:

            url = page_data['URL_function'][0]

            if page_data['Lang'][0] == 'EN':
                #limit page_data_en to 10 columns below
                page_data_en = page_data[["Comment", "Date", "Status",  "What's wrong", "Lookup_tags", 'Tags confirmed', 'Yes/No', 'Lookup_page_title', 'Lookup_group_EN', 'Lookup_group_FR']]
                page_data_en = page_data_en[page_data_en.Status != 'Spam']
                page_data_en = page_data_en[page_data_en.Status != 'Ignore']
                page_data_en = page_data_en[page_data_en.Status != 'Duplicate']
                page_data_en = page_data_en.reset_index(drop=True)
                #set nulls to [none]
                page_data_en['Lookup_group_EN'].fillna('[None]', inplace=True)
                #adds commas in between each character
                page_data_en['Lookup_page_title'] = [','.join(map(str, l)) for l in page_data_en['Lookup_page_title']]

                #set variable & convert to string
                group_link = (page_data_en['Lookup_group_EN'][0])
                group_link = ' '.join(map(str, group_link))

                if lang == 'en':
                    group_name = page_data_en['Lookup_group_EN'][0]
                    group_name = ' '.join(map(str, group_name))

                if lang == 'fr':
                    group_name = page_data_en['Lookup_group_FR'][0]
                    group_name = ' '.join(map(str, group_name))
                #grab page title
                title = page_data_en['Lookup_page_title'][0]

                #yes_no for all period
                #grab relevant columns
                yes_no_db = yes_no_db[["url", "yesno", "problemDate", "problem"]]
                #Fix urls
                yes_no_db['url'] = yes_no_db['url'].str.replace('/content/canadasite', 'www.canada.ca')
                yes_no_db['url'] = yes_no_db['url'].str.replace('www.canada.ca', 'https://www.canada.ca')
                yes_no_db['url'] = yes_no_db['url'].str.replace('https://https://', 'https://')
                #grab current page only and reset indexes
                yes_no_db = yes_no_db.loc[yes_no_db['url'] == page]
                yes_no_db = yes_no_db.reset_index(drop=True)

                #Parse date to Y-M-D
                counter = 0
                for index in yes_no_db['problemDate']:
                    arrWords = index.split(" ")
                    index = arrWords[3] + "-" + monthDict[arrWords[1]] + "-" + arrWords[2] 
                    yes_no_db['problemDate'][counter] = index
                    counter += 1

                #90 days db
                yes_no_db = yes_no_db[yes_no_db['problemDate'] >= earliest]
                #period selected db
                yes_no_period = yes_no_db[yes_no_db['problemDate'] <= end_date]
                yes_no_period = yes_no_db[yes_no_db['problemDate'] >= start_date]

                #90 days reasons
                reasons = yes_no_db[['problem']]
                #period selected reasons
                reasons_period = yes_no_period[['problem']]
                #Grabs yes/no & sorts by date, resets indexes, renames columns
                yes_no = yes_no_db[['problemDate', 'yesno']]
                yes_no = yes_no.rename(columns={"problemDate": "Date", "yesno": "Yes/No"})
                yes_no = yes_no.dropna()
                yes_no = yes_no.sort_values(by = 'Date', ascending=False)
                yes_no = yes_no.reset_index(drop=True)

                df_yes = (yes_no.groupby("Date")["Yes/No"]
                    .value_counts()
                    .unstack(fill_value=0)
                    .rename_axis(columns=None)
                    .eval("Percentage = Yes / (Yes + No)")
                    )


                df_yes= df_yes.reset_index(drop=False)
                df_yes = df_yes[['Date', 'Yes', 'No', 'Percentage']]
                df_yes= df_yes.reset_index(drop=True)

                df_yes = df_yes.sort_values(by = 'Date')
                df_yes= df_yes.reset_index(drop=True)

                #add column rolling mean
                df_yes['Rolling mean'] = df_yes.iloc[:,3].rolling(window=7).mean()
                #dates list and dates_r list
                dates = list(df_yes['Date'])
                dates_r = dates[::-1]
                #yes list and yes list reversed
                chart_yes = list(df_yes['Yes'])
                chart_yes = chart_yes[::-1]
                #no list and no list reversed
                chart_no = list(df_yes['No'])
                chart_no = chart_no[::-1]
                #Generate seperate lists of percentage & rolling mean
                daily_values = list(df_yes['Percentage'])
                weekly_values = list(df_yes['Rolling mean'])
                #new list but rounded too nearest 2 decimal places
                daily_perc = ["%.2f" % number for number in daily_values]
                weekly_perc = ["%.2f" % number for number in weekly_values]
                #new reversed lists
                daily_perc_r = daily_perc[::-1]
                weekly_perc_r = weekly_perc[::-1]

                start_plot = start_date
                end_plot = end_date

                #if less than dates range set to index 0
                if start_plot < dates[0]:
                    start_plot = dates[0]
                #if greater than dates range set to last index
                if end_plot > dates[-1]:
                    end_plot = dates[-1]

                all_start = dates[0]
                all_end = dates[-1]

                #Generates graph
                img = io.BytesIO()
                x = dates
                y1 = daily_values
                y2 = weekly_values
                fig, ax = plt.subplots()
                plt.ylim(0, 1)
                if lang == 'en':
                    ax.plot(x, y1, linewidth=0.5, label='Daily value')
                    ax.plot(x, y2, linewidth=3.0, label='Weekly rolling mean')
                    plt.title('Ratio of people who said they found their answer')

                if lang == 'fr':
                    ax.plot(x, y1, linewidth=0.5, label='Valeur quotidienne')
                    ax.plot(x, y2, linewidth=3.0, label='Moyenne mobile sur 7 jours')
                    plt.title('Proportion de gens qui disent avoir trouver leur réponse')

                plt.axvspan(start_plot, end_plot, color='blue', alpha=0.3)
                plt.legend()
                fig.autofmt_xdate()
                #x axis labels every 7 days
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
                #Grab number of yes/no to calculate yes percentage 90 days
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
                    score = (yes/ ( yes +  no))
                    score = format(score, '.2f')


                #replace nulls with false
                page_data_en["What's wrong"].fillna(False, inplace=True)
                page_data_en["Tags confirmed"].fillna(False, inplace=True)

                all_data_en = page_data_en.copy()

                #limit page_data to period
                page_data_en = page_data_en[page_data_en['Date'] <= end_date]
                page_data_en = page_data_en[page_data_en['Date'] >= start_date]

                #Generate list of date and yes/nos for selected period
                yes_no_period = yes_no_period[['problemDate', 'yesno']]
                yes_no_period = yes_no_period.rename(columns={"problemDate": "Date", "yesno": "Yes/No"})
                yes_no_period = yes_no_period.dropna()
                yes_no_period = yes_no_period.sort_values(by = 'Date', ascending=False)
                yes_no_period = yes_no_period.reset_index(drop=True)


                if yes_no_period.empty:
                    score_period = 'unavailable'
                    yes_period = 'unavailable'
                    no_period = 'unavailable'
                    delta = 'unavailable'
                #Grab number of yes/no to calculate yes percentage of period selected.
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

                    score_period = (yes_period / ( yes_period +  no_period))
                    score_period = format(score_period, '.2f')

                    #Calculate score difference
                    if score_period > score:
                        delta = '+' + format(float(score_period)-float(score), '.2f')
                    elif score_period < score:
                        delta =  format(float(score_period)-float(score), '.2f')
                    else:
                        if lang == 'en':
                            delta = 'no change'
                        if lang == 'fr':
                            delta = 'aucun changement'


                #by what's wrong reason
                #Grabs reasons counts and generates a dataframe for 90 days
                reasons[["problem"]] = reasons[["problem"]].replace([''], ['None'])
                reasons_db = reasons["problem"].value_counts()
                by_reason = reasons_db.to_frame()
                by_reason.columns = ['Feedback count']
                by_reason.reset_index(level=0, inplace=True)
                by_reason = by_reason[['Feedback count', 'index']]

                #Grabs reasons counts and generates a dataframe for selected period
                reasons_period[["problem"]] = reasons_period[["problem"]].replace([''], ['None'])
                reasons_period_db = reasons_period["problem"].value_counts()
                by_reason_period = reasons_period_db.to_frame()
                by_reason_period.columns = ['Feedback count']
                by_reason_period.reset_index(level=0, inplace=True)
                by_reason_period = by_reason_period[['Feedback count', 'index']]

                #drop nulls
                all_data_en = all_data_en.dropna()

                if all_data_en.empty:

                    if lang == 'en':
                        return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()))

                    if lang == 'fr':
                        return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()))

                else:
                    #create tags column
                    all_data_en['tags'] = [','.join(map(str, l)) for l in all_data_en['Lookup_tags']]

                    #remove the Lookup_tags column (it's not needed anymore)
                    all_data_en = all_data_en.drop(columns=['Lookup_tags'])
                    #dataf of all tags seperated (3 max?)
                    all_tags_en = all_data_en["tags"].str.split(",", n = 3, expand = True)
                    #add tags to alldata
                    all_data_en = all_data_en.join(all_tags_en)
                    all_data_en = all_data_en.drop(columns=['tags'])

                    #organizes and orders tags
                    tag_count = all_tags_en.apply(pd.Series.value_counts)
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

                    tag_dico_columns = ['Date', 'Comment']

                    for tag in unique_tags:
                      tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                    for tag, topic_df_en in all_data_en.groupby(0):
                      tag_dico[tag] = topic_df_en[['Date', 'Comment']]


                    if 1 in all_data_en.columns:
                        for tag, topic_df_en in all_data_en.groupby(1):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])

                    if 2 in all_data_en.columns:
                        for tag, topic_df_en in all_data_en.groupby(2):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])


                    for tag in tag_dico:
                        tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                    tag_dates = {}

                    for tag in tag_dico:
                      tag_dates[tag] = tag_dico[tag]['Date'].value_counts()



                    date_range = all_data_en['Date']
                    date_range = date_range.sort_values()
                    date_range = date_range.reset_index(drop=True)

                    for tag in tag_dates:
                      idx = pd.date_range(date_range.iloc[0], date_range.iloc[-1])
                      tag_dates[tag].index = pd.DatetimeIndex(tag_dates[tag].index)
                      tag_dates[tag] = tag_dates[tag].reindex(idx, fill_value=0)

                    #generate graph for tags
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
                    #drop columns and replace nulls with false
                    page_data_en = page_data_en.drop(columns=['Status'])
                    page_data_en = page_data_en.drop(columns=['Yes/No'])
                    page_data_en["What's wrong"].fillna(False, inplace=True)
                    page_data_en["Tags confirmed"].fillna(False, inplace=True)

                    page_data_en = page_data_en.dropna()


                    #get most frequent words for all of page
                    #get all words in a list parse to string and generate string of all words (all_words_en)
                    word_list_en = page_data_en["Comment"].tolist()
                    word_list_en = [str(i) for i in word_list_en]
                    all_words_en = ' '.join([str(elem) for elem in word_list_en])

                    #tokenize words
                    tokenizer = nltk.RegexpTokenizer(r"\w+")
                    tokens_en = tokenizer.tokenize(all_words_en)
                    #generate list of all words
                    words_en = []
                    for word in tokens_en:
                            words_en.append(word.lower())

                    #remove English stop words to get most frequent words
                    nltk.download('stopwords')
                    sw = nltk.corpus.stopwords.words('english')
                    sw.append('covid')
                    sw.append('19')
                    #generate list of words and make sure they are all characters
                    words_ns_en = []
                    for word in words_en:
                            if word not in sw and word.isalpha():
                                words_ns_en.append(word)
                    #get most common words
                    from nltk import FreqDist
                    fdist1 = FreqDist(words_ns_en)
                    most_common = fdist1.most_common(15)
                    mc = pd.DataFrame(most_common, columns =['Word', 'Count'])
                    mc = mc[['Count', 'Word']]


                    #Generate WordCloud from words_ns_en
                    if not words_ns_en:
                        cloud_url = ""

                    else:
                        cloud_words = " ".join(words_ns_en)
                        img = io.BytesIO()
                        wordcloud = WordCloud(background_color='white', max_font_size = 100, width=600, height=300).generate(cloud_words)
                        plt.figure(figsize=(10,5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        plt.show()
                        plt.savefig(img, format='png')
                        plt.close()
                        img.seek(0)

                        cloud_url = base64.b64encode(img.getvalue()).decode()



                    unconfirmed_en = page_data_en.loc[page_data_en['Tags confirmed'] == False]

                    #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
                    confirmed_en = page_data_en.loc[page_data_en['Tags confirmed'] == True]


                    if confirmed_en.empty and unconfirmed_en.empty :


                        if lang == 'en':
                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)


                    elif confirmed_en.empty:

                        unconfirmed_en = unconfirmed_en.reset_index(drop=True)
                        unconfirmed_en['tags'] = [','.join(map(str, l)) for l in unconfirmed_en['Lookup_tags']]
                        unconfirmed_en = unconfirmed_en.drop(columns=['Lookup_tags'])
                        unconfirmed_en = unconfirmed_en.drop(columns=['Tags confirmed'])
                        unconfirmed_en = unconfirmed_en.reset_index(drop=True)
                        unconfirmed_tags_en = unconfirmed_en["tags"].str.split(",", n = 3, expand = True)
                        unconfirmed_en = unconfirmed_en.join(unconfirmed_tags_en)
                        unconfirmed_en = unconfirmed_en.drop(columns=['tags'])

                        unconfirmed_tag_count = unconfirmed_tags_en.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Comment']

                        for tag in unconfirmed_unique_tags:
                          unconfirmed_tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_en in unconfirmed_en.groupby(0):
                          unconfirmed_tag_dico[tag] = topic_df_en[['Date', 'Comment']]


                        if 1 in unconfirmed_en.columns:
                            for tag, topic_df_en in unconfirmed_en.groupby(1):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_en[['Date', 'Comment']])

                        if 2 in unconfirmed_en.columns:
                            for tag, topic_df_en in unconfirmed_en.groupby(2):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_en[['Date', 'Comment']])


                        for tag in unconfirmed_tag_dico:
                            unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        unconfirmed_dict = { key: unconfirmed_tag_dico[key] for key in unconfirmed_unique_tags }

                        tag_columns = ['Date', 'Comment']

                        unconfirmed_tag_plots = { tag: tag_plots[tag] for tag in unconfirmed_unique_tags }
                        unconfirmed_plots = list(unconfirmed_tag_plots.values())

                        if lang == 'en':

                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common,  zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)


                    elif unconfirmed_en.empty:


                        #remove the Lookup_page_title column (it's not needed anymore)
                        confirmed_en = confirmed_en.drop(columns=['Tags confirmed'])

                        #resets the index for each row - needed for further processing
                        confirmed_en = confirmed_en.reset_index(drop=True)

                        #split dataframe for French comments - same comments as above for each line

                        #get data for specific page

                        #split tags and expand
                        confirmed_en['tags'] = [','.join(map(str, l)) for l in confirmed_en['Lookup_tags']]
                        tags_en = confirmed_en["tags"].str.split(",", n = 3, expand = True)
                        confirmed_en = confirmed_en.join(tags_en)
                        confirmed_en = confirmed_en.drop(columns=['tags'])



                        confirmed_en = confirmed_en.reset_index(drop=True)



                        #count the number for each tag
                        tag_count = tags_en.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Comment']

                        for tag in unique_tags:
                          tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_en in confirmed_en.groupby(0):
                          tag_dico[tag] = topic_df_en[['Date', 'Comment']]


                        if 1 in confirmed_en.columns:
                            for tag, topic_df_en in confirmed_en.groupby(1):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])

                        if 2 in confirmed_en.columns:
                            for tag, topic_df_en in page_data_en.groupby(2):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])


                        for tag in tag_dico:
                            tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        over_tags = by_tag[(by_tag > 3).any(1)]
                        under_tags = by_tag[(by_tag <= 3).any(1)]

                        over_unique_tags = list(over_tags.index)
                        under_unique_tags = list(under_tags.index)

                        over_dict = { key: tag_dico[key] for key in over_unique_tags }
                        under_dict = { key: tag_dico[key] for key in under_unique_tags }

                        tag_columns = ['Date', 'Comment']


                        over_tag_plots = { tag: tag_plots[tag] for tag in over_unique_tags }
                        over_plots = list(over_tag_plots.values())


                        under_tag_plots = { tag: tag_plots[tag] for tag in under_unique_tags }
                        under_plots = list(under_tag_plots.values())

                        #split feedback by tag

                        if lang == 'en':

                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common,  zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)


                    else:

                        confirmed_en = confirmed_en.drop(columns=['Tags confirmed'])

                        #resets the index for each row - needed for further processing
                        confirmed_en = confirmed_en.reset_index(drop=True)

                        #split dataframe for French comments - same comments as above for each line

                        #get data for specific page

                        #split tags and expand
                        confirmed_en['tags'] = [','.join(map(str, l)) for l in confirmed_en['Lookup_tags']]
                        tags_en = confirmed_en["tags"].str.split(",", n = 3, expand = True)
                        confirmed_en = confirmed_en.join(tags_en)
                        confirmed_en = confirmed_en.drop(columns=['tags'])






                        confirmed_en = confirmed_en.reset_index(drop=True)



                        #count the number for each tag
                        tag_count = tags_en.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Comment']

                        for tag in unique_tags:
                          tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_en in confirmed_en.groupby(0):
                          tag_dico[tag] = topic_df_en[['Date', 'Comment']]


                        if 1 in confirmed_en.columns:
                            for tag, topic_df_en in confirmed_en.groupby(1):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])

                        if 2 in confirmed_en.columns:
                            for tag, topic_df_en in page_data_en.groupby(2):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])


                        for tag in tag_dico:
                            tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        over_tags = by_tag[(by_tag > 3).any(1)]
                        under_tags = by_tag[(by_tag <= 3).any(1)]

                        over_unique_tags = list(over_tags.index)
                        under_unique_tags = list(under_tags.index)

                        over_dict = { key: tag_dico[key] for key in over_unique_tags }
                        under_dict = { key: tag_dico[key] for key in under_unique_tags }

                        tag_columns = ['Date', 'Comment']


                        over_tag_plots = { tag: tag_plots[tag] for tag in over_unique_tags }
                        over_plots = list(over_tag_plots.values())


                        under_tag_plots = { tag: tag_plots[tag] for tag in under_unique_tags }
                        under_plots = list(under_tag_plots.values())


                        unconfirmed_en = unconfirmed_en.reset_index(drop=True)
                        unconfirmed_en['tags'] = [','.join(map(str, l)) for l in unconfirmed_en['Lookup_tags']]
                        unconfirmed_en = unconfirmed_en.drop(columns=['Lookup_tags'])
                        unconfirmed_en = unconfirmed_en.drop(columns=['Tags confirmed'])
                        unconfirmed_en = unconfirmed_en.reset_index(drop=True)
                        unconfirmed_tags_en = unconfirmed_en["tags"].str.split(",", n = 3, expand = True)
                        unconfirmed_en = unconfirmed_en.join(unconfirmed_tags_en)
                        unconfirmed_en = unconfirmed_en.drop(columns=['tags'])

                        unconfirmed_tag_count = unconfirmed_tags_en.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Comment']

                        for tag in unconfirmed_unique_tags:
                          unconfirmed_tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_en in unconfirmed_en.groupby(0):
                          unconfirmed_tag_dico[tag] = topic_df_en[['Date', 'Comment']]


                        if 1 in unconfirmed_en.columns:
                            for tag, topic_df_en in unconfirmed_en.groupby(1):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_en[['Date', 'Comment']])

                        if 2 in unconfirmed_en.columns:
                            for tag, topic_df_en in unconfirmed_en.groupby(2):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_en[['Date', 'Comment']])


                        for tag in unconfirmed_tag_dico:
                            unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        unconfirmed_dict = { key: unconfirmed_tag_dico[key] for key in unconfirmed_unique_tags }

                        tag_columns = ['Date', 'Comment']

                        unconfirmed_tag_plots = { tag: tag_plots[tag] for tag in unconfirmed_unique_tags }
                        unconfirmed_plots = list(unconfirmed_tag_plots.values())

                        if lang == 'en':

                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common,  zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags,  list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)


            #process to follow if French
            else:

                #keep only relevant columns from the dataframe
                page_data_fr = page_data[["Comment", "Date", "Status",  "What's wrong", "Lookup_FR_tag", 'Tags confirmed', 'Yes/No', 'Lookup_page_title', 'Lookup_group_EN', 'Lookup_group_FR']]
                page_data_fr = page_data_fr[page_data_fr.Status != 'Spam']
                page_data_fr = page_data_fr[page_data_fr.Status != 'Ignore']
                page_data_fr = page_data_fr[page_data_fr.Status != 'Duplicate']
                page_data_fr = page_data_fr.reset_index(drop=True)

                page_data_fr['Lookup_page_title'] = [','.join(map(str, l)) for l in page_data_fr['Lookup_page_title']]
                page_data_fr['Lookup_group_EN'] = [','.join(map(str, l)) for l in page_data_fr['Lookup_group_EN']]
                page_data_fr['Lookup_group_FR'] = [','.join(map(str, l)) for l in page_data_fr['Lookup_group_FR']]

                group_link = page_data_fr['Lookup_group_EN'][0]

                if lang == 'en':
                    group_name = page_data_fr['Lookup_group_EN'][0]

                if lang == 'fr':
                    group_name = page_data_fr['Lookup_group_FR'][0]

                title = page_data_fr['Lookup_page_title'][0]

                yes_no_db = yes_no_db[["url", "yesno", "problemDate", "problem"]]
                yes_no_db['url'] = yes_no_db['url'].str.replace('/content/canadasite', 'www.canada.ca')
                yes_no_db['url'] = yes_no_db['url'].str.replace('www.canada.ca', 'https://www.canada.ca')
                yes_no_db['url'] = yes_no_db['url'].str.replace('https://https://', 'https://')
                yes_no_db = yes_no_db.loc[yes_no_db['url'] == page]
                yes_no_db = yes_no_db.reset_index(drop=True)

                counter = 0
                for index in yes_no_db['problemDate']:
                    arrWords = index.split(" ")
                    index = arrWords[3] + "-" + monthDict[arrWords[1]] + "-" + arrWords[2] 
                    yes_no_db['problemDate'][counter] = index
                    counter += 1

                yes_no_db = yes_no_db[yes_no_db['problemDate'] >= earliest]

                yes_no_period = yes_no_db[yes_no_db['problemDate'] <= end_date]
                yes_no_period = yes_no_db[yes_no_db['problemDate'] >= start_date]

                reasons = yes_no_db[['problem']]
                reasons_period = yes_no_period[['problem']]

                yes_no = yes_no_db[['problemDate', 'yesno']]
                yes_no = yes_no.rename(columns={"problemDate": "Date", "yesno": "Yes/No"})
                yes_no = yes_no.dropna()
                yes_no = yes_no.sort_values(by = 'Date', ascending=False)
                yes_no = yes_no.reset_index(drop=True)

                df_yes = (yes_no.groupby("Date")["Yes/No"]
                    .value_counts()
                    .unstack(fill_value=0)
                    .rename_axis(columns=None)
                    .eval("Percentage = Yes / (Yes + No)")
                    )


                df_yes= df_yes.reset_index(drop=False)
                df_yes = df_yes[['Date', 'Yes', 'No', 'Percentage']]
                df_yes= df_yes.reset_index(drop=True)

                df_yes = df_yes.sort_values(by = 'Date')
                df_yes= df_yes.reset_index(drop=True)

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
                plt.ylim(0, 1)
                if lang == 'en':
                    ax.plot(x, y1, linewidth=0.5, label='Daily value')
                    ax.plot(x, y2, linewidth=3.0, label='Weekly rolling mean')
                    plt.title('Ratio of people who said they found their answer')

                if lang == 'fr':
                    ax.plot(x, y1, linewidth=0.5, label='Valeur quotidienne')
                    ax.plot(x, y2, linewidth=3.0, label='Moyenne mobile sur 7 jours')
                    plt.title('Proportion de gens qui disent avoir trouver leur réponse')

                plt.axvspan(start_plot, end_plot, color='blue', alpha=0.3)
                plt.legend()
                fig.autofmt_xdate()
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
                    score = (yes/ ( yes +  no))
                    score = format(score, '.2f')

                page_data_fr["What's wrong"].fillna(False, inplace=True)
                page_data_fr["Tags confirmed"].fillna(False, inplace=True)

                all_data_fr = page_data_fr.copy()


                page_data_fr = page_data_fr[page_data_fr['Date'] <= end_date]
                page_data_fr = page_data_fr[page_data_fr['Date'] >= start_date]


                yes_no_period = yes_no_period[['problemDate', 'yesno']]
                yes_no_period = yes_no_period.rename(columns={"problemDate": "Date", "yesno": "Yes/No"})
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

                    score_period = (yes_period / ( yes_period +  no_period))
                    score_period = format(score_period, '.2f')
                    if score_period > score:
                        delta = '+' + format(float(score_period)-float(score), '.2f')
                    elif score_period < score:
                        delta =  format(float(score_period)-float(score), '.2f')
                    else:
                        if lang == 'en':
                            delta = 'no change'
                        if lang == 'fr':
                            delta = 'aucun changement'


                #by what's wrong reason
                reasons[["problem"]] = reasons[["problem"]].replace([''], ['Aucune'])
                reasons[["problem"]] = reasons[["problem"]].replace(["I'm not in the right place"], ["Je ne suis pas au bon endroit"])
                reasons[["problem"]] = reasons[["problem"]].replace(["Other reason"], ["Autre raison"])
                reasons[["problem"]] = reasons[["problem"]].replace(["The information isn't clear"], ["L'information n'est pas claire"])
                reasons[["problem"]] = reasons[["problem"]].replace(["Something is broken or incorrect"], ["Quelque chose est brisé ou incorrect"])
                reasons[["problem"]] = reasons[["problem"]].replace(["The answer I need is missing"], ["La réponse dont j'ai besoin n'est pas là"])

                reasons_db = reasons["problem"].value_counts()
                by_reason = reasons_db.to_frame()
                by_reason.columns = ['Feedback count']
                by_reason.reset_index(level=0, inplace=True)
                by_reason = by_reason[['Feedback count', 'index']]


                reasons_period[["problem"]] = reasons_period[["problem"]].replace([''], ['Aucune'])
                reasons_period[["problem"]] = reasons_period[["problem"]].replace(["I'm not in the right place"], ["Je ne suis pas au bon endroit"])
                reasons_period[["problem"]] = reasons_period[["problem"]].replace(["Other reason"], ["Autre raison"])
                reasons_period[["problem"]] = reasons_period[["problem"]].replace(["The information isn't clear"], ["L'information n'est pas claire"])
                reasons_period[["problem"]] = reasons_period[["problem"]].replace(["Something is broken or incorrect"], ["Quelque chose est brisé ou incorrect"])
                reasons_period[["problem"]] = reasons_period[["problem"]].replace(["The answer I need is missing"], ["La réponse dont j'ai besoin n'est pas là"])

                reasons_period_db = reasons_period["problem"].value_counts()
                by_reason_period = reasons_period_db.to_frame()
                by_reason_period.columns = ['Feedback count']
                by_reason_period.reset_index(level=0, inplace=True)
                by_reason_period = by_reason_period[['Feedback count', 'index']]

                all_data_fr = all_data_fr.dropna()

                if all_data_fr.empty:

                    if lang == 'en':
                        return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()))

                    if lang == 'fr':
                        return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()))

                else:


                    all_data_fr['tags'] = [','.join(map(str, l)) for l in all_data_fr['Lookup_FR_tag']]

                    #remove the Lookup_tags column (it's not needed anymore)
                    all_data_fr = all_data_fr.drop(columns=['Lookup_FR_tag'])
                    all_tags_fr = all_data_fr["tags"].str.split(",", n = 3, expand = True)
                    all_data_fr = all_data_fr.join(all_tags_fr)
                    all_data_fr = all_data_fr.drop(columns=['tags'])

                    tag_count = all_tags_fr.apply(pd.Series.value_counts)
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

                    tag_dico_columns = ['Date', 'Comment']

                    for tag in unique_tags:
                      tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                    for tag, topic_df_fr in all_data_fr.groupby(0):
                      tag_dico[tag] = topic_df_fr[['Date', 'Comment']]


                    if 1 in all_data_fr.columns:
                        for tag, topic_df_fr in all_data_fr.groupby(1):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])

                    if 2 in all_data_fr.columns:
                        for tag, topic_df_fr in all_data_fr.groupby(2):
                            if tag_dico[tag].empty:
                                tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                            else:
                                tag_dico[tag] = tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])


                    for tag in tag_dico:
                        tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                    tag_dates = {}

                    for tag in tag_dico:
                      tag_dates[tag] = tag_dico[tag]['Date'].value_counts()



                    date_range = all_data_fr['Date']
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
                      start_plot = start_date
                      end_plot = end_date
                      all_start = dates[0]
                      all_end = dates[-1]
                      img = io.BytesIO()
                      x = dates
                      y1 = daily_values
                      y2 = weekly_values
                      column = tag_dates[tag]['Count']
                      high_y = column.max()
                      max_y = high_y + 5
                      fig, ax = plt.subplots()
                      if lang == 'en':
                          ax.bar(x, y1, color=(0.2, 0.4, 0.6, 0.6), linewidth=0.5, label='Daily value')
                          ax.plot(x, y2, color='black', linewidth=3.0, label='Weekly rolling mean')
                          plt.title(tag + '\n' + 'Number of commments per day')

                      if lang == 'fr':
                          ax.bar(x, y1, color=(0.2, 0.4, 0.6, 0.6), linewidth=0.5, label='Valeur quotidienne')
                          ax.plot(x, y2, color='black', linewidth=3.0, label='Moyenne mobile sur 7 jours')
                          plt.title(tag + '\n' + 'Nombre de commentaires par jour')

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

                    page_data_fr = page_data_fr.drop(columns=['Status'])
                    page_data_fr = page_data_fr.drop(columns=['Yes/No'])
                    page_data_fr["What's wrong"].fillna(False, inplace=True)
                    page_data_fr["Tags confirmed"].fillna(False, inplace=True)

                    page_data_fr = page_data_fr.dropna()


                    #get most frequent words for all of page
                    #get all words in a list
                    word_list_fr = page_data_fr["Comment"].tolist()
                    word_list_fr = [str(i) for i in word_list_fr]
                    all_words_fr = ' '.join([str(elem) for elem in word_list_fr])

                    #tokenize words
                    tokenizer = nltk.RegexpTokenizer(r"\w+")
                    tokens_fr = tokenizer.tokenize(all_words_fr)
                    words_fr = []
                    for word in tokens_fr:
                            words_fr.append(word.lower())


                    #remove English stop words to get most frequent words
                    nltk.download('stopwords')
                    sw = nltk.corpus.stopwords.words('french')
                    sw.append('covid')
                    sw.append('19')
                    sw.append('a')
                    sw.append('si')
                    sw.append('avoir')
                    sw.append('savoir')
                    sw.append('combien')
                    sw.append('être')
                    sw.append('où')
                    sw.append('comment')
                    sw.append('puis')
                    sw.append('peuvent')
                    sw.append('fait')
                    sw.append('aucun')
                    sw.append('bonjour')
                    sw.append('depuis')
                    sw.append('chez')
                    sw.append('faire')
                    sw.append('peut')
                    sw.append('plus')
                    sw.append('veux')
                    sw.append('dois')
                    sw.append('doit')
                    sw.append('dit')
                    sw.append('merci')
                    sw.append('cela')
                    sw.append('pouvons')
                    sw.append('pouvaient')
                    sw.append('vers')

                    words_ns_fr = []
                    for word in words_fr:
                            if word not in sw and word.isalpha():
                                words_ns_fr.append(word)


                    #get most common words
                    from nltk import FreqDist
                    fdist1 = FreqDist(words_ns_fr)
                    most_common = fdist1.most_common(15)
                    mc = pd.DataFrame(most_common, columns =['Mots', 'Nombre'])
                    mc = mc[['Nombre', 'Mots']]


                    #Generate WordCloud
                    if not words_ns_fr:
                        cloud_url = ""

                    else:
                        cloud_words = " ".join(words_ns_fr)
                        img = io.BytesIO()
                        wordcloud = WordCloud(background_color='white', max_font_size = 100, width=600, height=300).generate(cloud_words)
                        plt.figure(figsize=(10,5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        plt.show()
                        plt.savefig(img, format='png')
                        plt.close()
                        img.seek(0)

                        cloud_url = base64.b64encode(img.getvalue()).decode()

                    #get unconfirmed tags

                    unconfirmed_fr = page_data_fr.loc[page_data_fr['Tags confirmed'] == False]


                    confirmed_fr = page_data_fr.loc[page_data_fr['Tags confirmed'] == True]



                    if confirmed_fr.empty and unconfirmed_fr.empty :

                        if lang == 'en':
                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, zip=zip, group_link = group_link, group_name = group_name, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                    elif confirmed_fr.empty:

                        unconfirmed_fr = unconfirmed_fr.reset_index(drop=True)
                        unconfirmed_fr['tags'] = [','.join(map(str, l)) for l in unconfirmed_fr['Lookup_FR_tag']]
                        unconfirmed_fr = unconfirmed_fr.drop(columns=['Lookup_FR_tag'])
                        unconfirmed_fr = unconfirmed_fr.drop(columns=['Tags confirmed'])
                        unconfirmed_fr = unconfirmed_fr.reset_index(drop=True)
                        unconfirmed_tags_fr = unconfirmed_fr["tags"].str.split(",", n = 3, expand = True)
                        unconfirmed_fr = unconfirmed_fr.join(unconfirmed_tags_fr)
                        unconfirmed_fr = unconfirmed_fr.drop(columns=['tags'])

                        unconfirmed_tag_count = unconfirmed_tags_fr.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Comment']

                        for tag in unconfirmed_unique_tags:
                          unconfirmed_tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_fr in unconfirmed_fr.groupby(0):
                          unconfirmed_tag_dico[tag] = topic_df_fr[['Date', 'Comment']]


                        if 1 in unconfirmed_fr.columns:
                            for tag, topic_df_fr in unconfirmed_fr.groupby(1):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])

                        if 2 in unconfirmed_fr.columns:
                            for tag, topic_df_fr in unconfirmed_fr.groupby(2):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])


                        for tag in unconfirmed_tag_dico:
                            unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        unconfirmed_dict = { key: unconfirmed_tag_dico[key] for key in unconfirmed_unique_tags }

                        tag_columns = ['Date', 'Comment']

                        unconfirmed_tag_plots = { tag: tag_plots[tag] for tag in unconfirmed_unique_tags }
                        unconfirmed_plots = list(unconfirmed_tag_plots.values())

                        if lang == 'en':

                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common,  zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)


                    elif unconfirmed_fr.empty:


                        confirmed_fr = confirmed_fr.drop(columns=['Tags confirmed'])

                        #resets the index for each row - needed for further processing
                        confirmed_fr = confirmed_fr.reset_index(drop=True)

                        #split dataframe for French comments - same comments as above for each line

                        #get data for specific page

                        #split tags and expand
                        confirmed_fr['tags'] = [','.join(map(str, l)) for l in confirmed_fr['Lookup_FR_tag']]
                        tags_fr = confirmed_fr["tags"].str.split(",", n = 3, expand = True)
                        confirmed_fr = confirmed_fr.join(tags_fr)
                        confirmed_fr = confirmed_fr.drop(columns=['tags'])




                        confirmed_fr = confirmed_fr.reset_index(drop=True)



                        #count the number for each tag
                        tag_count = tags_fr.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Commentaire']

                        for tag in unique_tags:
                          tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_fr in confirmed_fr.groupby(0):
                          tag_dico[tag] = topic_df_fr[['Date', 'Comment']]


                        if 1 in page_data_fr.columns:
                            for tag, topic_df_fr in confirmed_fr.groupby(1):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])

                        if 2 in page_data_fr.columns:
                            for tag, topic_df_fr in confirmed_fr.groupby(2):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])


                        for tag in tag_dico:
                            tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        over_tags = by_tag[(by_tag > 3).any(1)]
                        under_tags = by_tag[(by_tag <= 3).any(1)]

                        over_unique_tags = list(over_tags.index)
                        under_unique_tags = list(under_tags.index)

                        over_dict = { key: tag_dico[key] for key in over_unique_tags }
                        under_dict = { key: tag_dico[key] for key in under_unique_tags }

                        over_tag_plots = { tag: tag_plots[tag] for tag in over_unique_tags }
                        over_plots = list(over_tag_plots.values())

                        under_tag_plots = { tag: tag_plots[tag] for tag in under_unique_tags }
                        under_plots = list(under_tag_plots.values())
                        column_names = ['Nombre de rétroactions', 'Étiquette', 'Mots significatifs']

                        if lang == 'en':

                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common,  zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)


                    else:



                        #split dataframe for French comments - same comments as above for each line

                        #get data for specific page

                        #split tags and expand

                        confirmed_fr = confirmed_fr.drop(columns=['Tags confirmed'])

                        #resets the index for each row - needed for further processing
                        confirmed_fr = confirmed_fr.reset_index(drop=True)

                        #split dataframe for French comments - same comments as above for each line

                        #get data for specific page

                        #split tags and expand
                        confirmed_fr['tags'] = [','.join(map(str, l)) for l in confirmed_fr['Lookup_FR_tag']]
                        tags_fr = confirmed_fr["tags"].str.split(",", n = 3, expand = True)
                        confirmed_fr = confirmed_fr.join(tags_fr)
                        confirmed_fr = confirmed_fr.drop(columns=['tags'])


                        confirmed_fr = confirmed_fr.reset_index(drop=True)



                        #count the number for each tag
                        tag_count = tags_fr.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Commentaire']

                        for tag in unique_tags:
                          tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_fr in confirmed_fr.groupby(0):
                          tag_dico[tag] = topic_df_fr[['Date', 'Comment']]


                        if 1 in page_data_fr.columns:
                            for tag, topic_df_fr in confirmed_fr.groupby(1):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])

                        if 2 in page_data_fr.columns:
                            for tag, topic_df_fr in confirmed_fr.groupby(2):
                                if tag_dico[tag].empty:
                                    tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])


                        for tag in tag_dico:
                            tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        over_tags = by_tag[(by_tag > 3).any(1)]
                        under_tags = by_tag[(by_tag <= 3).any(1)]

                        over_unique_tags = list(over_tags.index)
                        under_unique_tags = list(under_tags.index)

                        over_dict = { key: tag_dico[key] for key in over_unique_tags }
                        under_dict = { key: tag_dico[key] for key in under_unique_tags }

                        over_tag_plots = { tag: tag_plots[tag] for tag in over_unique_tags }
                        over_plots = list(over_tag_plots.values())

                        under_tag_plots = { tag: tag_plots[tag] for tag in under_unique_tags }
                        under_plots = list(under_tag_plots.values())
                        column_names = ['Nombre de rétroactions', 'Étiquette', 'Mots significatifs']


                        unconfirmed_fr = unconfirmed_fr.reset_index(drop=True)
                        unconfirmed_fr['tags'] = [','.join(map(str, l)) for l in unconfirmed_fr['Lookup_FR_tag']]
                        unconfirmed_fr = unconfirmed_fr.drop(columns=['Lookup_FR_tag'])
                        unconfirmed_fr = unconfirmed_fr.drop(columns=['Tags confirmed'])
                        unconfirmed_fr = unconfirmed_fr.reset_index(drop=True)
                        unconfirmed_tags_fr = unconfirmed_fr["tags"].str.split(",", n = 3, expand = True)
                        unconfirmed_fr = unconfirmed_fr.join(unconfirmed_tags_fr)
                        unconfirmed_fr = unconfirmed_fr.drop(columns=['tags'])

                        unconfirmed_tag_count = unconfirmed_tags_fr.apply(pd.Series.value_counts)
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

                        tag_dico_columns = ['Date', 'Comment']

                        for tag in unconfirmed_unique_tags:
                          unconfirmed_tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

                        for tag, topic_df_fr in unconfirmed_fr.groupby(0):
                          unconfirmed_tag_dico[tag] = topic_df_fr[['Date', 'Comment']]


                        if 1 in unconfirmed_fr.columns:
                            for tag, topic_df_fr in unconfirmed_fr.groupby(1):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])

                        if 2 in unconfirmed_fr.columns:
                            for tag, topic_df_fr in unconfirmed_fr.groupby(2):
                                if unconfirmed_tag_dico[tag].empty:
                                    unconfirmed_tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                                else:
                                    unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])


                        for tag in unconfirmed_tag_dico:
                            unconfirmed_tag_dico[tag] = unconfirmed_tag_dico[tag].sort_values(by = 'Date', ascending=False)


                        unconfirmed_dict = { key: unconfirmed_tag_dico[key] for key in unconfirmed_unique_tags }

                        tag_columns = ['Date', 'Comment']

                        unconfirmed_tag_plots = { tag: tag_plots[tag] for tag in unconfirmed_unique_tags }
                        unconfirmed_plots = list(unconfirmed_tag_plots.values())

                        if lang == 'en':

                            return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common,  zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

                        if lang == 'fr':
                            return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), list = list, tag_columns = tag_columns, yes_period = yes_period, no_period = no_period, score_period = score_period, all_start = all_start, all_end = all_end, over_tags = zip(over_unique_tags, list(over_tags['Feedback count'].values.tolist()), over_plots, over_unique_tags), over_dict = over_dict, under_tags = zip(under_unique_tags, list(under_tags['Feedback count'].values.tolist()), under_plots, under_unique_tags), under_dict = under_dict, delta = delta, lang = lang, chart_columns = chart_columns, daily_perc_r = daily_perc_r, weekly_perc_r = weekly_perc_r, dates_r = dates_r, chart_yes = chart_yes, chart_no = chart_no, unconfirmed_tags = zip(unconfirmed_unique_tags, list(unconfirmed_by_tag['Feedback count'].values.tolist()), unconfirmed_plots, unconfirmed_unique_tags), unconfirmed_dict = unconfirmed_dict, group_link = group_link, group_name = group_name, row_data_reason_period = list(by_reason_period.values.tolist()), cloud_url = cloud_url)

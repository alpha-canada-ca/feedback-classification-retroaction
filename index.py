def index():

    from flask import Flask
    from flask import request
    from flask import app, render_template
    import requests

    lang = request.args.get('lang', 'en')

    if lang == 'en':
        return render_template("index_en.html", lang=lang)

    if lang == 'fr':
        return render_template("index_fr.html", lang=lang)



    #split feedback by What's wrongfully

    #get most meaningful word by what's wrong

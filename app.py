
from flask import Flask
from flask import request
from flask import app, render_template
import index
import page_index
import by_page
import by_group
import group_index

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return index.index()


@app.route('/bypage', methods=['GET', 'POST'])

def bypage():
    return by_page.bypage()

@app.route('/bygroup', methods=['GET', 'POST'])
def bygroup():
    return by_group.bygroup()


@app.route('/group_index', methods=['GET', 'POST'])
def groupindex():
    return group_index.groupindex()

@app.route('/page_index', methods=['GET', 'POST'])
def pageindex():
    return page_index.pageindex()

if __name__ == '__main__':
    app.run()

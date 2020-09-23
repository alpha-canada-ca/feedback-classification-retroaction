
from flask import Flask
from flask import request
from flask import app, render_template
import index
import by_page

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return index.index()


@app.route('/bypage', methods=['GET', 'POST'])

def bypage():
    return by_page.bypage()


if __name__ == '__main__':
    app.run()

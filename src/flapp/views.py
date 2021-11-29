import pandas as pd
import pickle
import urllib

from flask import render_template
from flask import request
from src.flapp import app




@app.route('/')
@app.route('/index')
def homepage():
    return render_template("index.html")

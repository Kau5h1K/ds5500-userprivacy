from flask import Flask
app = Flask(__name__)
app.jinja_env.filters['zip'] = zip
from src.flapp import routes
from flask import Flask, request, Response, json
import numpy as np
import pandas as pd


# load house data
df = pd.read_csv("./data/house_price.csv", header=None)


# create flask app instance
app = Flask(__name__)
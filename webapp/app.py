from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def predict():
    print("outcome tab")
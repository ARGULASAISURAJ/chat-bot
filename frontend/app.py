from flask import Flask, request, Response, render_template, jsonify
import json
import string
from os import listdir
import os
import logging
from logging.handlers import RotatingFileHandler
import random


class Chatbot:
    def __init__(self):
        pass

app = Flask(__name__)
chatbot = Chatbot()

@app.route('/')
def home():
    return render_template("consent.html")

@app.route('/panas')
def panas():
    return render_template("panas.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/bot')
def bot():
    return render_template("bot.html")


if __name__ == "__main__":
    try:
        #app.debug = True
        app.run(host='34.75.26.239',port=3000)
    finally:
        print("Exit")

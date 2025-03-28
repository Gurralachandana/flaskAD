#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Load the trained model
model_path = "churn_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        data = request.args.to_dict()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return str(prediction[0])  # Return plain text
    except Exception as e:
        return str(e)  # Return error as plain text

if __name__ == "__main__":
    app.run(debug=True)


import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
BHP_model = pickle.load(open('regmodel.pkl', 'rb'))
scalar= pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    transformed_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    prediction = BHP_model.predict(transformed_data)
    print(prediction[0])
    return jsonify(prediction[0])

if __name__=="__main__":
    app.run(debug=True)   
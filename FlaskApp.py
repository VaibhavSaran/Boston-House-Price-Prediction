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

@app.route('/predict',methods=['POST'])
def predict():
    form_data= [float(x) for x in request.form.values()]
    final_data = scalar.transform(np.array(form_data).reshape(1,-1))
    print(final_data)
    predicted_value = BHP_model.predict(final_data)[0]
    return render_template("home.html",prediction_text="The predicted price is: {}".format(predicted_value))


if __name__=="__main__":
    app.run(debug=True)   
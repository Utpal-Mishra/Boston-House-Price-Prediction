import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,app,jsonify,url_for,render_template

app=Flask(__name__)

## Load the model
Model  = pickle.load(open('RegModel.pkl','rb'))
Scalar = pickle.load(open('Scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/PredictAPI',methods=['POST'])
def PredictAPI():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))

    Data = Scalar.transform(np.array(list(data.values())).reshape(1,-1))

    Output = Model.predict(Data)
    print(Output[0])

    return jsonify(Output[0])

@app.route('/Predict',methods=['POST'])
def Predict():
    Data=[float(x) for x in request.form.values()]

    Input = Scalar.transform(np.array(Data).reshape(1,-1))
    print(Input)

    output = Model.predict(Input)[0]

    return render_template("home.html", prediction_text="Predicted Price: {}".format(round(output, 3)))


if __name__=="__main__":
    app.run(debug=True)
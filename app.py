import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__) # define the application

Model = pickle.load(open('RegModel.pkl', 'rb')) # Load
Scaling = pickle.load(open('Scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/PredictAPI', methods = ['POST'])


def PredictAPI():
    data = request.json['data'] # data extracted from json file will be in key, value pair
    print(data)

    print(np.array(list(data.values())).reshape(1, -1))
    NewData = Scaling.transform(np.array(list(data.values())).reshape(1, -1))

    Output = Model.predict(NewData)
    
    return jsonify(Output[0])

if __name__ == "__main__":
    app.run(debug =  True)
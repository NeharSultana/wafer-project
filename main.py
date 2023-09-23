from wsgiref import simple_server
from flask import Flask, render_template, request, Response
from flask_cors import CORS, cross_origin
from logger import App_Logger
from dataPreProcessing import Data_Preprocessing
from modelPreProcessing import Model_Preprocessing
import os
import json

app=Flask(__name__)
app.debug = True
CORS(app)

@app.route("/", methods = ['GET'])
@cross_origin()
def home():
    file = open("Logs/prediction_log.txt", "a+")
    log_writer = App_Logger()
    log_writer.log(file, "Start")
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
@cross_origin()
def predict_data():
    predict_val = Data_Preprocessing("Batch_files/Prediction_Batch_Files", "predict")
    predict_val.preprocess()
    predict_model = Model_Preprocessing("predict")
    path,json_predictions = predict_model.predict()
    return Response("Prediction File created at !!!" + str(path) + 'and few of the predictions are ' + str(json.loads(json_predictions)))

def train_data():
    train_val = Data_Preprocessing("Batch_files/Training_Batch_Files", "train")
    train_val.preprocess()
    train_model = Model_Preprocessing("train")
    train_model.trainModel()

if __name__ == "__main__":
    app.run(debug=True)
    #train_data()
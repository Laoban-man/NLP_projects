import pickle
import os
from flask import Flask, render_template, request,url_for
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from random import randint

app = Flask(__name__)

@app.route('/')
def home_index():
    full_filenames=[]
    full_filenames.append(os.path.join("static",'img', 'emergency.png'))
    full_filenames.append(os.path.join("static",'img', 'eda.jpeg'))
    full_filenames.append(os.path.join("static",'img','performance.jpeg'))
    full_filenames.append(os.path.join("static",'img', 'binoculars.png'))
    return render_template("index.html",user_image = full_filenames)

@app.route('/eda', methods=['GET',"POST"])
def eda_index():
    return render_template("eda2.html")

@app.route('/performance', methods=['GET'])
def performance_index():

    file_path=os.path.join("data","train.csv")
    data=pd.read_csv(file_path)
    result=pickle.load(open("data/location.pkl","rb"))
    new_result=pd.DataFrame(result)
    new_result.replace("Italia","None",inplace=True)
    data["new_location"]=new_result

    data_sample=data[data["keyword"].notna()].reset_index(drop=True)
    final_data=pd.concat([data_sample,pd.get_dummies(data_sample["new_location"])],axis=1)
    final_data=pd.concat([data_sample,pd.get_dummies(data_sample["keyword"],prefix="keyword")],axis=1)
    y=final_data["target"]

    data=pickle.load(open("data/final_data.pkl","rb"))
    data=data.iloc[:,:-2]
    rfc= pickle.load(open("models/rfc.sav","rb"))
    lgb= pickle.load(open("models/lgb.sav","rb"))
    tnn = tf.keras.models.load_model("models/tnn")
    X_train,X_test,y_train,y_test=tts(data,y,train_size=0.8)
    ypred_test=rfc.predict(X_test)
    ypred_train=rfc.predict(X_train)
    rfc_test=accuracy_score(y_test,ypred_test)
    rfc_train=accuracy_score(y_train,ypred_train)

    ypred_test=lgb.predict(X_test)
    ypred_train=lgb.predict(X_train)
    lgb_test=accuracy_score(y_test,ypred_test)
    lgb_train=accuracy_score(y_train,ypred_train)

    ypred_test=(tnn.predict(X_test)>=0.5)*1
    ypred_train=(tnn.predict(X_train)>=0.5)*1
    tnn_test=accuracy_score(y_test,ypred_test)
    tnn_train=accuracy_score(y_train,ypred_train)
#    bert = tf.keras.models.load_model("models/bert")
    return render_template("performance.html",data=[[rfc_test,rfc_train],[lgb_test,lgb_train],[tnn_test,tnn_train]])

@app.route('/predictions', methods=['GET'])
def predictions_index():
    file_path=os.path.join("data","test.csv")
    data=pd.read_csv(file_path)
    rownb=randint(0, len(data["text"]))
    sample=data["text"][rownb]

    data=pickle.load(open("data/final_data.pkl","rb"))
    data=data.iloc[:,:-2]
    sample_vec=np.array(data.iloc[rownb,:]).reshape(1,-1)
    rfc= pickle.load(open("models/rfc.sav","rb"))
    lgb= pickle.load(open("models/lgb.sav","rb"))
    tnn = tf.keras.models.load_model("models/tnn")

    rfc_pred=rfc.predict(sample_vec)

    lgb_pred=lgb.predict(sample_vec)

    tnn_pred=(tnn.predict(sample_vec)>=0.5)*1

    return render_template("predictions.html",data=[sample,[rfc_pred,lgb_pred,tnn_pred]])


if __name__ == "__main__":
    app.debug = True
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True)

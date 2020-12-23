import pickle
from flask import Flask, render_template, request
import requests


app = Flask(__name__)



@app.route('/', methods=['GET'])
def home_index():
	return render_template("index.html")

@app.route('/eda/', methods=['GET'])
def eda_index():
	return render_template("eda.html")

@app.route('/performance/', methods=['GET'])
def performance_index():
	return render_template("performance.html")

@app.route('/predictions/', methods=['GET'])
def predictions_index():
	return render_template("predictions.html")


@app.route('/predict/', methods=['POST'])
def result():

	r = requests.post("http://127.0.0.1:8000/model/predict/", json={
		})
	api_response = r.json()

	return render_template("prediction.html", price=api_response['reg'])

if __name__ == '__main__':
    app.debug = True
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True)

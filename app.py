import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
count_vector = pickle.load(open('count_vector.pkl', 'rb'))


@app.route('/home')
def home():
    return render_template('index_1.html')


@app.route('/Proceed')
def Proceed():
    return render_template('form.html')


@app.route('/LogisticRegression')
def LogisticRegression():
    return render_template('LogisticRegression.html')


@app.route('/predict', methods=['POST'])
def predict():
    Open = int(request.form['Open'])
    Volume = int(request.form['Volume'])
    High = int(request.form['High'])
    Low = int(request.form['Low'])
    output = model.predict([[Volume,Open, High, Low]])
    if output == 1:
        res_val = "Positive"
    elif output == -1:
        res_val = "Negative"
    else:
        res_val = "Rating which does not match with"

    return render_template('LogisticRegression.html', prediction_text='Expected Price of Stock: {}'.format(res_val))


@app.route('/movingaverage')
def movingaverage():
    return render_template('MovingAverage.html')


@app.route('/dailyreturns')
def dailyreturns():
    return render_template('DailyReturns.html')


@app.route('/NavieBayes')
def NavieBayes():
    return render_template('NavieBayes.html')


@app.route('/predict_2', methods=['POST'])
def predict_2():
    news = str(request.form['news'])

    text = count_vector.transform([news])
    output = model1.predict(text)
    if output == 1:
        res_val = "Positive"
    elif output == 0:
        res_val = "Negative"
    else:
        res_val = "Rating which does not match with"

    return render_template('NavieBayes.html', prediction_text='Customer has given a {} review'.format(res_val))


@app.route('/NeuralNetworks')
def NeuralNetworks():
    return render_template('NeuralNetwork.html')



if __name__ == "__main__":
    app.run()

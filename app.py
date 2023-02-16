
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import re

#load model
filename = 'trained_pipeline.pkl'
trained_pipline = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

classes = ["Arabic",
           "Danish",
           "Dutch",
           "English",
           "French",
           "German",
           "Greek",
           "Hindi",
           "Italian",
           "Kannada",
           "Malayalam",
           "Portugeese",
           "Russian",
           "Spanish",
           "Sweedish",
           "Tamil",
           "Turkish"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #text = request.form['text']
        #text = [text]
        int_features = [x for x in request.form.values()]
        text = int_features[0]
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
        text = re.sub(r"[[]]", " ", text)
        text = text.lower()
        pred = trained_pipline.predict([text])
        language_predicted = classes[pred[0]]
        return render_template('index.html', prediction=language_predicted)  



if __name__ == '__main__':
	app.run(debug=True) 


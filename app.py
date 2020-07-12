
#o/n: When you run this script a web app or web server will be started which can be accessed
#from any browser.
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask app
app = Flask(__name__)
#load pickled model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')#o/n: go to the root directory i.e web application directory

#rendering html
def home():
    return render_template('index.html')

################################################################################################################
#PREDICTION APIs
@app.route('/predict',methods=['POST'])#'/predict' means set separate a sepate URL for the app. 'POST' - to submit data to be processed to the server. in addition, 'GET'-  to request data from the server.
######API 1######
#Reads input from the  user intrerface, push them thru model and outputs prediction
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)#'prediction[0]means take the 1st value of the predictions

    return render_template('index.html', prediction_text=' The predicted House price is :  $ {}'.format(output))

#DIRECT API FUNCTION. Can use a json file for the inputs and return prediction
@app.route('/predict_api',methods=['POST'])

######API 2######
#function for direct API call through request
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
################################################################################################################

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
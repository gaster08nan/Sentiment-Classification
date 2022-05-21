import os
from flask import Flask, render_template, request

import model_v2 as model
import utils as ut
from config import Settings
config = Settings()

app = Flask(__name__)


app = Flask(__name__)

#app.config['UPLOAD_FOLDER'] = os.path.join('static','images')

@app.route("/")
def home_page():
    #img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    tmp_sequence = [request.form['text']]
    #load token
    tokeninzer = ut.load_pickle(config.model_path, 'Token')
    select = request.form.get('models')
    if "RNN" in select:
        clfs = model.init_RNN_Classifier()
        model_name = '142999_LSTM_1.h5'
    elif "GRU" in select:
        clfs = model.init_GRU_Classifier()
        model_name = '143000_GRU_1.h5'
    elif "bi" in select:
        clfs = model.init_BiLSTM_Classifier()
        model_name = '143000_bi-LSTM_1.h5'
    else:
        clfs = model.init_LSTM_Classifier()
        model_name = '142999_LSTM_1.h5'

    #preprocess data for test:
    test_sequence = clfs.predict_preprocessing(tmp_sequence, tokeninzer, 
                                                max_leng = 500, min_leng = 10)
    #weight path
    weight_path = os.path.join(config.model_path,model_name)
    input_dim = int(model_name.split('_')[0])
    #load mode
    clfs.load_weights(weight_path ,input_dim = input_dim)
    #clfs.summary()
    #predict
    pred = clfs.predict(test_sequence)
    if pred[0] >= 0.5:
        result = 'Possitive'
        possitive = True
    else:
        result = 'Negative'
        possitive = False
    return render_template('predict.html', text_value = request.form['text'],
                                        predict_value =  pred[0])


if __name__ == '__main__':
    app.run(debug=True, port = 5000)
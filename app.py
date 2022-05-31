import os
from flask import Flask, render_template, request
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import model_v2 as model
import utils as ut
from config import Settings
config = Settings()

app = Flask(__name__)

#app.config['UPLOAD_FOLDER'] = os.path.join('static','images')


@app.route("/")
def home_page():
    """
    Load home page (index.html) as default page

    Returns:
        _type_: remder index.html when call function
    """
    #img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    """
    Get selected value from drop down menu to load model, get review from textbox, preprocess and predict the sentimen

    Returns:
        _type_: render predict.html with predict value to show predict result in the modal
    """
    # get the review from the textbox
    tmp_sequence = [request.form['text']]
    # get the selected value from the dropbox to load model
    select = request.form.get('models')
    # flag use to load weight or entire model for the final model
    load_weight = True
    if "Final" in select:  # if the Final Model is selected
        # intit the model type
        clfs = model.init_RNN_Classifier()
        # init the model name
        model_name = '/version 3'
        # for the final model, load entire model instead of weight
        load_weight = False
    elif "GRU" in select:  # load GRU model
        clfs = model.init_GRU_Classifier()
        model_name = '143000_1024_128_GRU.h5'
    elif "Bi" in select:  # load Bidirectional LSTM model
        clfs = model.init_BiLSTM_Classifier()
        model_name = '143000_128_32_BiLSTM.h5'
    else:  # load LSTM model
        clfs = model.init_LSTM_Classifier()
        model_name = '142999_1024_64_LSTM.h5'

    if load_weight:  # if we need to load weight only
        # load the tokenizer
        tokeninzer = ut.load_pickle(config.model_path, 'Token')
        # preprocess data for test:
        test_sequence = clfs.predict_preprocessing(tmp_sequence, tokeninzer,
                                                   max_leng=500, min_leng=10)
        # weight path
        weight_path = os.path.join(config.model_path, model_name)
        input_dim = int(model_name.split('_')[0])
        output_dim = int(model_name.split('_')[1])
        node = int(model_name.split('_')[2])
        # load mode
        clfs.load_weight(weight_path, input_dim=input_dim,
                         output_dim=output_dim, node=node)
        # clfs.summary()
        # predict
        pred = clfs.predict(test_sequence)

    else:  # load entire model instead of weight
        # load the tokenizer
        tokenizer = ut.load_pickle(config.model_path, 'version 3.tokenizer')
        # preprocess the data
        x_test = ut.preprocess_text_for_final_model(tmp_sequence, tokenizer)
        # load model
        clfs.load_model(config.model_path + model_name)
        # predict
        pred = clfs.predict(x_test)

    return render_template('predict.html', text_value=request.form['text'],
                           predict_value=pred[0])


if __name__ == '__main__':
    app.run(debug=True, port=5000)

import numpy as np
import pandas as pd
import utils as ut
import model_v2 as model
import os
from config import Settings

config = Settings()


def train():
    """
    Load and preprocess the training data, train the model with the init structure, save the weight to the Model folder.
    """
    # load data:
    config = Settings()
    data_path = os.path.join(config.data_folder, config.data_file_name)
    raw_data = ut.load_data(data_path, '\t')
    reviews = raw_data.review
    labels = raw_data.sentiment

    # preprocess training data
    clfs = model.init_LSTM_Classifier()
    process_x, label, token = clfs.preprocessing(
        reviews, labels, init_max_len=500, init_min_len=50)
    x_train, x_test, y_train, y_test = ut.train_test_split(
        process_x, label, test_size=0.4, random_state=42)

    input_dim = len(ut.create_vocab(token))+1
    weight_path = os.path.join(config.model_path, f'{input_dim}_weight.h5')
    # define model
    clfs.build_model(input_dim)
    # train model
    history = clfs.train(
        x_train, y_train, epochs=config.epochs, batch_size=config.batch_size)
    # save weight
    clfs.save_weights(weight_path)
    print(f'Save weight to: {weight_path}')
    # save train process:
    clfs.plot_training_process(history, save_path=config.model_path)


def main():
    history = train()


if __name__ == "__main__":
    main()

import os
import numpy as np
import keras
import matplotlib.pyplot as plt


from keras.models import Sequential, load_model
from keras.layers import Dense
#from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
#from keras.callbacks import ModelCheckpoint, EarlyStopping

import utils as ut

def init_LSTM_Classifier():
    class LSTM_cls():
        def __init__(self)-> None:
            super().__init__()
            self.model = None

        def build_model(self, input_dim = 1000):
            """
            Build LSTM model
            """
            model = Sequential()
            # Add an Embedding layer expecting input vocab of size 1000, and
            # output embedding dimension of size 1024.
            model.add(Embedding(input_dim=input_dim, output_dim=1024))

            # Add a LSTM layer with 64 internal units.
            model.add(LSTM(64))
            # Add a Dense layer with 1 units.
            model.add(Dense(1, activation = 'sigmoid'))

            #model.summary()
            model.compile(
                optimizer="Adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
                )
            self.model = model
            return self.model

        def preprocessing(self, reviews, label, old_tokenizer = None, init_max_len = 500, init_min_len = 10):
            """
            Prerocess data for training:
                *lower case letters, remove punctuation and html tags
                *remove to short or to long sequence
                *padding sequence
            Input:
                revires (list): list of training reviews
                init_max_len: maximum length of sequence
                init_min_len: minimum length of sequence
            Output:
                padded_seq (np array): array of padded sequences
                new_label (np arrray): array of label
                tokenizer: tokenizer
            """
            #remove puncuation and html tags, lower case letters
            clean_data = ut.cleaning(reviews)
            #create tokenizer, change sequences to int values
            tokenized_seq, seq_len, tokenizer = ut.tokenize(clean_data, old_tokenizer)
            #Remove sequences have length not meet the requires length
            new_tokenized_seq, new_seq_len, new_label = ut.remove_seq(
                                                                        tokenized_seq,
                                                                        seq_len, label,
                                                                        max_len = init_max_len,
                                                                        min_len = init_min_len
                                                                    )
            max_leng = max(new_seq_len)
            #add padding to sequence to meet maximun length
            padded_seq, _ = ut.padding_sequence(
                                                            new_tokenized_seq,
                                                            new_seq_len,
                                                            max_len = max_leng
                                                        )
            padded_seq = np.array(padded_seq)
            new_label = np.array(new_label)
            return padded_seq, new_label, tokenizer

        def predict_preprocessing(self, reviews, tokeninzer, max_leng, min_leng):
            """
            Preprocess data for predict:
                * lower case letters, remove html tags and punctuation
                * tokenize predict sequences and change to int values
                * remove predict sequences to long or to short
            Input:
                reviews (list): predict sequence
                token: tokenizer use when training
                max_leng: maximum sequences length use in training
                min_leng: minimum sequences lenght
            Output:
                processed_seq (np array): preprocessed sequences
            """
            clean_data = ut.cleaning(reviews)
            pred_seq = tokeninzer.texts_to_sequences(clean_data)
            seq_leng = [len(x) for x in pred_seq]
            label = [-1 for x in range(len(pred_seq))]

            rm_seq, new_seq_leng, _ = ut.remove_seq(
                                                            pred_seq, seq_leng, label,
                                                            max_len = max_leng,
                                                            min_len = min_leng
                                                        )
            processed_seq, processed_seq_len = ut.padding_sequence(
                                                                    rm_seq,
                                                                    new_seq_leng,
                                                                    max_len = max_leng
                                                                )

            return np.array(processed_seq)

        def summary(self):
            """
            Smmary model architecture
            """
            self.model.summary()

        def train(
                    self, data, labels,
                    val_split = 0.2,
                    epochs = 12,
                    batch_size = 64,
                    callbacks = None
                ):
            """
            Tranin model and return history
            """
            history = self.model.fit(
                            data,
                            labels,
                            validation_split = val_split,
                            epochs = epochs,
                            batch_size = batch_size,
                            callbacks = callbacks
                        )
            return history

        def predict(self, x_pred):
            """
            Predict list of sequence
            Input:
                x_pred (list): list contain processed predict sequence
            Output:
                array of predict values
            """
            return self.model.predict(x_pred)

        def load_model(self, filepath):
            """
            Load model structure and weight
            """
            self.model = load_model(filepath)

        def save_model(self, filepath):
            """
            Save model structure and weight
            """
            return self.model.save(filepath)

        def save_weights(self, weight_path):
            """
            Save model weight
            """
            self.model.save_weights(weight_path)

        def load_weights(self, weight_path, input_dim = 1000):
            """
            Define model architecture and load model weight
            """
            self.model = self.build_model(input_dim)
            self.model.load_weights(weight_path)

        def plot_training_process(self, history, show = False, save = True, save_path='.'):
            """
            Plot training process of model
            Input:
                history: training history
                show (bool): show figure if True
                save (bool): save figure if True
            Output:
                None
            """
            fig, axes= plt.subplots(nrows= 1, ncols= 2)
            fig.set_figwidth(20)
            fig.set_figheight(8)
            fig.suptitle("Training Progress", fontsize= 30)

            axes[0].plot(history.history['accuracy'])
            axes[0].plot(history.history['val_accuracy'])
            axes[0].set_title('model accuracy', fontsize= 15)
            axes[0].set_xlabel('epoch')
            axes[0].set_ylabel('accuracy')
            axes[0].legend(['train', 'test'], loc='upper left')

            axes[1].plot(history.history['loss'])
            axes[1].plot(history.history['val_loss'])
            axes[1].set_title('model loss', fontsize= 15)
            axes[1].set_xlabel('epoch')
            axes[1].set_ylabel('loss')
            axes[1].legend(['train', 'test'], loc='upper left')
            if save:
                save_name = os.path.join(save_path, "training_process.png")
                plt.savefig(save_name)
            if show:
                plt.show()

    return LSTM_cls()

if __name__ == "__main__":
    model = init_LSTM_Classifier()

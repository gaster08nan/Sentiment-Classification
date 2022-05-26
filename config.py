from pydantic import BaseSettings


class Settings(BaseSettings):
    data_folder = "Data"
    data_file_name = "labeledTrainData.tsv"
    test_data_file_name = 'testData.tsv'
    base_url = 'https://www.imdb.com/title/tt0816692/reviews?ref_=tt_urv'
    model_path = "Model"
    checkpoint_path = "Checkpoint"
    tokenizer_path = model_path
    epochs = 15
    batch_size = 64
    html_path = "templates"
    model_type = ['RNN', 'GRU', 'LSTM', 'BiLSTM', 'BERT']


if __name__ == "__main__":
    Settings()

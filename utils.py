from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import os
import re
import pickle as pkl
import string
from urllib.parse import urljoin
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests


from bs4 import BeautifulSoup as soup
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

string.punctuation = string.punctuation + '“'+'”'+'-'+'’'+'‘'+'—'+'¨'
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


def create_folder(folder_path):
    """
    Create folder if not exist in folder path
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created: ", folder_path)
    else:
        print(f"{folder_path} exists")


def load_data(data_path, sep=','):
    """
    Read data form data_path
    """

    if ".csv" in data_path:
        data = pd.read_csv(data_path, sep=',')
    elif ".tsv" in data_path:
        data = pd.read_csv(data_path, sep='\t')
    else:
        data = pd.read_csv(data_path, sep=f'{sep}')

    return data


def plot_data_len_distribution(seq_data, start=0, end=-1):
    """
    Plot sequence's length distribution
    """

    seq_lens = [len(x) for x in seq_data]
    check_point = list(range(10, max(seq_lens), 10))
    tmp = pd.DataFrame(seq_lens, columns=['Frequen'])
    data_distribution = []

    for i in range(len(check_point[:-1])):
        dwn = sum(tmp.Frequen <= check_point[i])
        up = sum(tmp.Frequen <= check_point[i+1])
        data_distribution.append(up - dwn)

    # visualize data distribution
    # Figure Size
    fig = plt.figure(figsize=(25, 15))

    # Horizontal Bar Plot
    #plt.bar(x, y)
    plt.bar(check_point[start+1:end], data_distribution[start:end], width=10)
    plt.title('Data distribution', fontweight='bold', fontsize=20)
    plt.xlabel('Length of sequence', fontweight='bold', fontsize=15)
    plt.ylabel('Number of sequence', fontweight='bold', fontsize=15)

    # Show Plot
    plt.show()


def cleanhtml(raw_html):
    """
    remove html tags in reviews
    """

    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


def cleaning(raw_data):
    """
    Return review to lower case, remove html tags, remove punctuation
    Input:
        raw_data (pandas.Series): raw review
    Output:
        process_x (list): preprocess review list
    """

    clean_data = []
    for review in raw_data:
        # To lower case
        review = review.lower()
        # Remove html tag
        review = cleanhtml(review)
        # Remove punctuation
        new_review = "".join(
            [char for char in review if char not in string.punctuation])
        clean_data.append(new_review)

    return clean_data


def remove_seq(seq2int, seq_len, label, max_len=500, min_len=10, index=None):
    """
    Truncating sequence under min_len and above max_len
    Input:
        seq2int: sequence
        seq_len: sequences's length
        label: label of sequences
    Output:
        new_seq2int: new sequences
        new_seq_len: new sequences's length
        new_seq_label: new labels
    """

    # remove sequences have lengh < 50 and > 500
    new_seq2int = []
    new_seq_len = []
    new_seq_label = []
    new_index = []
    for i in range(len(seq_len)):
        if seq_len[i] <= max_len and seq_len[i] >= min_len:
            new_seq2int.append(seq2int[i])
            new_seq_len.append(seq_len[i])
            new_seq_label.append(label[i])
            # save index in the list, use to predict test
            if index is not None:
                new_index.append(i)
    if index is not None:
        return new_seq2int, new_seq_len, new_seq_label, new_index
    else:
        return new_seq2int, new_seq_len, new_seq_label


def tokenize(input_sequence, old_tokenizer=None):
    """
    Create vocab, tokenize sequence and save tokenizer
    Input:
        input_sequence: list of sequences
        old_tokenizer: tokenizer, if None mean don't have token and create new one, else mean aldready have tokenizer
    Output:
    """

    if old_tokenizer is None:
        tokenizer = Tokenizer(num_words=None, oov_token="<OOV>",)
        tokenizer.fit_on_texts(input_sequence)
    else:
        tokenizer = old_tokenizer
    seq2int = tokenizer.texts_to_sequences(input_sequence)
    seq_len = [len(x) for x in seq2int]

    return seq2int, seq_len, tokenizer


def create_vocab(token):
    """
    Create vocabulary form input token
    """

    # create vocab
    vocab = token.word_index

    return vocab


def preprocess_text(review):
    """ Convert text to lowercase, Remove HTML tags, non-letters and stop words

        Input: Raw text: String

        Output: Processed text: String """

    stop_words = stopwords.words("english")
    # clean HTML tags
    clean_html = soup(review).get_text()
    # Remove non-letters
    clean_non_letters = re.sub("[^a-zA-Z]", " ", clean_html)
    # Convert to lowercase
    cleaned_lowercase = clean_non_letters.lower()
    # Tokenize
    word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    words = word_tokenizer.tokenize(cleaned_lowercase)
    # Remove Stopword and Apply Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    processed_text = [lemmatizer.lemmatize(
        w) for w in words if w not in stop_words]
    return " ".join(processed_text)


def padding_sequence(sequence, seq_len, post=True, max_len=500):
    """
    Padding zeroes to sequence to meet the  require length
    Input:
        sequence (list): sequence need to padding
        seq_len (list): list contain the length of all sequence
        post (bool): if True, padding zeroes after the last sequence's value,
                    else padding before the first value
        max_len (int): maximun length of a sequence, 
                padding zeroes for the sequence to meet the max len
    Output:
        pad_seq (list): padded sequence need to padding
        seq_len (list): list contain the new length of all sequence
    """

    # padding:
    pad_seq = sequence.copy()
    for i in range(len(seq_len)):
        if seq_len[i] < max_len:
            padding = list(np.zeros(max_len - seq_len[i]))
            if post:
                pad_seq[i] += padding
            else:
                pad_seq[i] = padding + pad_seq
        seq_len[i] = len(pad_seq[i])

    return pad_seq, seq_len


def preprocess_text_for_final_model(input_sequence, tokenizer, padding_mode='pre', truncating_mode='pre', maxlen=1416):
    """
    Final text preprocessing use for final model, include: 
        lower case words, 
        remove punctuations, 
        html tags, stopword, 
        padding and truncating sequence, 
        tokenize the sequence 

    Args:
        input_sequence (list): list of sequence to preprocessing
        tokenizer (_type_): tokenizer use in training process
        padding_mode (str, optional): type of padding method, 'pre' mean padding to the begin of the sequence, 'post' mean padding at the end. 
            Defaults to 'pre'.
        truncating_mode (str, optional): type of truncating method, 'pre' mean truncating to the begin of the sequence, 'post' mean truncating at the end. 
            Defaults to 'pre'.
        maxlen (int, optional): Maximun lenght of sequence. Defaults to 1416.

    Returns:
        _type_: _description_
    """
    # to lower case words and remove html and punctuation
    process_data = [preprocess_text(p) for p in input_sequence]
    # tokenize the process data
    test_sequences = tokenizer.texts_to_sequences(process_data)
    test_padded = pad_sequences(
        test_sequences, padding=padding_mode, truncating=truncating_mode, maxlen=maxlen)

    return np.array(test_padded)


def save_pickle(path, filename, file):
    """
    Save file using pickle
    Input: 
        path (string): path to save folder
        filename (string): file name
        file: object need to save
    Output:
        none
    """

    with open(os.path.join(path, filename), "wb") as filesave:
        pkl.dump(file, filesave)


def load_pickle(path, filename):
    """
    Load file using pickle
    Input:
        path (string): path to save folder
        filename (string): file name
    Output:
        file: the loaded file
    """
    with open(os.path.join(path, filename), "rb") as filesave:
        file = pkl.load(filesave)

    return file


def save_joblib(path, filename, file):
    """
    Save file using joblib
    Input: 
        path (string): path to save folder
        filename (string): file name
        file: object need to save
    Output:
        none
    """

    joblib.dump(file, os.path.join(path, filename))


def load_joblib(path, filename):
    """
    Load file using joblib and return the object
    Input:
        path (string): path to save folder
        filename (string): file name
    Output:
        file: the loaded file
    """

    loaded_file = joblib.load(os.path.join(path, filename))

    return loaded_file


def split_train_test(data, labels, split_ratio=0.25):
    """
    Split data to training set and test set by split ratio
    Input:
        X (np array): data
        y (np array): label
        split_ratio (float): ratio of training set and test set
    Output:
        x_train (np array): training data
        x_test (np array): test data
        y_train (np array): traing labels
        y_test (np array): test labels
    """

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=split_ratio,
        random_state=42
    )

    return x_train, x_test, y_train, y_test


def get_review(soup_page, reviews=[], sentiments=[]):
    """
    Get review from pages getting from beautifulsoup, then append them to review list and setiments list
    The setiment is copmute base on the rating of that review (rating > 5 => sentiment =1, otherwise = 0)
    Input:
        soup_page: page from beautifulsoup
        reviews (list): first page review list
        sentiments (list): first page sentiments list
    Output:
        reviews (list): reviews_list
        sentiments (list): sentiments_list
    """
    html_page = soup_page.findAll('div', attrs={'class': 'review-container'})
    for i, review in enumerate(html_page):
        # Nếu mà không tìm thấy rating hoặc review thì bỏ qua
        try:
            user_review = review.find(
                'div', attrs={'class': 'text show-more__control'}).text.strip()
            tmp_span_list = review.find('span', attrs={'class': 'rating-other-user-rating'}
                                        ).findAll('span')
        except Exception as e:
            print("error: ", e)
            continue
        rating_score = int(tmp_span_list[0].text) / \
            int(tmp_span_list[-1].text.replace('/', ''))
        if rating_score >= 0.5:
            sentiments.append(1)
        else:
            sentiments.append(0)
        reviews.append(user_review)
    return reviews, sentiments


def crawl_data_from_url(base_url, no_load=1, reviews=[], sentiments=[]):
    """
    Crawl reviews in url, can't get review from the first pages
    Input:
        base_url (string): url from reviews page
        no_load (int): number of load more data
        reviews (list): first page review list
        sentiments (list): first page sentiments list
    Output:
        reviews (list): all users review the url
        sentiments (list): the sentimet of the reviews
    """
    # Remove ?ref in base urt
    if '?ref' in base_url:
        base_url = base_url.split('?ref')[0]

    return_reviews = reviews
    return_sentiments = sentiments

    # Get html page from the url
    respond = requests.get(base_url)
    session_page = soup(respond.content)

    ##
    # if load page 0 or both reviews and sentiments list in empty,
    # load the first page save to reviews and sentiments list
    ##
    if no_load == 0 or len(reviews) == 0:
        return_reviews, return_sentiments = get_review(session_page)

    # Find the data key for loading purpose
    data_key = session_page.find('div',
                                 attrs={'class': 'load-more-data'}
                                 ).attrs['data-key']
    # New url for loading
    new_url = f"{base_url}/_ajax?paginationKey={data_key}"

    for i in range(no_load):
        new_page = soup(requests.get(new_url).content)
        return_reviews, return_sentiments = get_review(new_page,
                                                       return_reviews,
                                                       return_sentiments)
        new_data_key = new_page.find('div',
                                     attrs={'class': 'load-more-data'}
                                     ).attrs['data-key']
        new_url = f"{base_url}/_ajax?paginationKey={new_data_key}"

    return return_reviews, return_sentiments

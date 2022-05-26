import os
import tkinter as tk
from tkinter import ttk
import pandas as pd

import utils as ut
import model_v2 as model
from config import Settings

config = Settings()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # root window
        self.title('Imdb Review Sentiments Classifier Demo')
        self.geometry('600x500')
        self.style = ttk.Style(self)
        # select theme
        self.style.theme_use('xpnative')

        # review text frame
        review_frame = tk.LabelFrame(self)
        review_frame.grid(column=0, row=0, padx=20, pady=20, sticky=tk.NS)
        # review frame - label
        review_lb = tk.Label(review_frame, text='Custom Review:')
        review_lb.grid(column=0, row=0, padx=10, pady=10,  sticky='w')
        # review frame - frame text
        review_txt = tk.Text(review_frame, height=15, width=40)
        # text place holder
        review_txt.insert('1.0', 'write your review here')
        review_txt.grid(column=0, row=1, padx=10, pady=10)

        # Define a function for switching the frames
        def change_to_main():
            review_frame.grid(column=0, row=0, padx=20, pady=20, sticky=tk.NS)
            result_frame.grid(column=1, row=0, padx=20, pady=20, sticky=tk.SE)
            crawl_frame.grid_forget()

        def change_to_crawl():
            crawl_frame.grid(column=0, row=0, padx=20, pady=20, sticky=tk.NS)
            review_frame.grid_forget()
            result_frame.grid_forget()

        # Function
        def classiffy_func():
            # get sequences
            text = review_txt.get('1.0', 'end')
            # Init model
            clfs = model.init_LSTM_Classifier()
            weight_path = 'Model//143000_weight.h5'
            input_dim = 143000
            # Load tokenizer
            tokenizer = ut.load_pickle(config.model_path, 'Token')
            seuqnce = [text]
            # Process test sequence
            test_sequence = clfs.predict_preprocessing(
                seuqnce,
                tokenizer,
                max_leng=500,
                min_leng=10
            )
            # Load model
            clfs.load_weights(weight_path, input_dim)
            predict_result = clfs.predict(test_sequence)
            if predict_result[0] <= 0.5:
                result_lb.config(text='Possitive',
                                 background='green', foreground='#ffffff')
            else:
                result_lb.config(
                    text='Negative', background='red', foreground='#ffffff')

        # change frame button
        review_change_frame_btn = tk.Button(
            review_frame, text='Change to crawl Data Frame', command=change_to_crawl)
        review_change_frame_btn.grid(column=0, row=2, padx=10, pady=10)

        # result frame
        result_frame = tk.LabelFrame(self, text='Result')
        result_frame.grid(column=1, row=0, padx=20, pady=20, sticky=tk.SE)
        # result frame - label
        result_lb = tk.Label(result_frame, text=review_txt.get('1.0', 'end'))
        result_lb.grid(column=0, row=0, padx=10, pady=10,  sticky='w')
        # result frame - button
        show_result_button = ttk.Button(
            result_frame, text='Classiffy', command=classiffy_func)
        show_result_button.grid(column=0, row=1, padx=10, pady=10)

        # crawl data frame
        crawl_frame = tk.LabelFrame(self)
        # label
        input_url_lbl = tk.Label(
            crawl_frame, text='Input url to crawl reviews')
        input_url_lbl.grid(column=0, row=0, padx=10, pady=10)
        # label
        load_page_lbl = tk.Label(crawl_frame, text='Number of pages')
        load_page_lbl.grid(column=1, row=0, padx=10, pady=10)
        # url input text
        input_url_txt = tk.Entry(crawl_frame, width=15,)
        input_url_txt.grid(column=0, row=1, padx=10, pady=10)
        # url input text
        load_page_txt = tk.Entry(crawl_frame, width=5,)
        load_page_txt.grid(column=1, row=1, padx=10, pady=10)
        # change frame button
        crawl_change_frame_btn = tk.Button(
            crawl_frame, text='Change to review Frame', command=change_to_main)
        crawl_change_frame_btn.grid(column=0, row=2, padx=10, pady=10)

        def crawl_review():
            url = input_url_txt.get()
            np_load = int(load_page_txt.get())
            return_reviews, return_sentiments = ut.crawl_data_from_url(
                url, np_load)
            df = pd.DataFrame({"Reviews": return_reviews,
                               "sentiment": return_sentiments})
            df.to_csv(os.path.join(config.data_folder, 'crawl_data.csv'))

        # crawl reviews button
        crawl_review_btn = tk.Button(
            crawl_frame, text='Crawl review', command=crawl_review)
        crawl_review_btn.grid(column=1, row=2, padx=10, pady=10)


if __name__ == "__main__":
    app = App()
    app.mainloop()

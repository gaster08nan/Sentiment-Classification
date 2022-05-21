import os
# import requests
# from bs4 import BeautifulSoup as soup
import pandas as pd

import utils as ut
from config import Settings

config = Settings()

def Crawl_data(base_url = config.base_url, save_path = config.data_folder, load_time = 3):

    return_reviews, return_sentiments = ut.crawl_data_from_url(
                                                                base_url, 
                                                                no_load = load_time, 
                                                                reviews = [], 
                                                                sentiments = []
                                                            )

    df = pd.DataFrame({"Reviews":return_reviews,
                    "sentiment":return_sentiments})
    file_name = f'{len(df)}_crawling_data.csv'
    df.save_csv(os.path.join(save_path, file_name))

def main()
{
    base_url = config.base_url
    save_path = config.data_folder
    load_time = 1

    Crawl_data(base_url, save_path, load_time)
}

if __name__ = "__main__":
    main()
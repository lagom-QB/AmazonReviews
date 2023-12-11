import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import fire

import streamlit as st

import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

def load_data(file_loc='assets/data1.csv', inputText=None, inputRating=None):
    # Check if file_loc exists
    if not os.path.exists(file_loc):
        st.write(f'{file_loc} not found at "asset"')
        raise FileNotFoundError(f'{file_loc} not found at "asset"')
    else:
        # st.write(f'File not found at "asset" ... Folder contains: {os.listdir("assets")}') # data.csv
        data = pd.read_csv(file_loc)
        num_cols = len(data.columns)
        data.columns = [f'column_{i}' for i in range(0, num_cols)]

        # if inputText is not None:
        #     # Create a new DataFrame using the inputText and inputRating
        #     new_data = pd.DataFrame({'sentiment': [inputText], 'rating': [inputRating]})
        #     st.write('Returning a row of data ....')
        #     return new_data
        # else:
        #     st.write('Returning 8% of the data ....')
        #     return data.sample(frac=0.08).reset_index(drop=True)
    
def load_data_1(file_loc='assets/data.csv', inputText=None, inputRating=None):
    # Check if file_loc exists
    if not os.path.exists(file_loc):
        st.write(f'File not found at {file_loc}')
        raise FileNotFoundError(f'File not found at {file_loc}')
    
    data = pd.read_csv(file_loc, names=[f'column_{i}' for i in range(0, 6)])

    if inputText is not None and inputRating is not None:
        # Create a new DataFrame using the inputText and inputRating
        new_data = pd.DataFrame({'sentiment': [inputText], 'rating': [inputRating]})
        st.write('Returning a row of data ....', new_data)
        return new_data
    else:
        st.write('Returning 80% of the data ....', data.head(5))
        return data.sample(frac=0.8).reset_index(drop=True)

def preprocess_data(data):

    return data

def get_sentiment(data):
    # if a column in data has string values, add it to the category_cols list else add it to the numerical_cols list 
    category_cols, numerical_cols = [], []
    for col in data.columns:
        if data[col].dtype == 'object':
            category_cols.append(col)
        else:
            numerical_cols.append(col)
    
    data['sentiment'] = data[category_cols].fillna('').agg(' '.join, axis=1)
    data['rating'] = data[numerical_cols].max(axis=1)

    return data[['sentiment', 'rating']]

def plot_ratings(data):
    plt.figure(figsize=(10,5))

    sns.countplot(y='rating', 
                 data=data, 
                 hue='rating', 
                 orient='v', 
                 edgecolor='black', 
                 linewidth=.1)
    
    plt.title('Distribution of Ratings', fontsize=20)
    plt.xlabel('Count', fontsize=10)
    plt.ylabel('Rating', fontsize=10)
    
    plt.show()

if __name__ == '__main__':
    fire.Fire(load_data)
    fire.Fire(preprocess_data)
    fire.Fire(get_sentiment)
    fire.Fire(plot_ratings)

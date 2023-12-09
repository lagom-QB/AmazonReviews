import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import fire

import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
    
def load_data(file_loc='assets/data.csv', inputText=None, inputRating=None):
    data = pd.read_csv(file_loc, names=[f'column_{i}' for i in range(0, 6)])
    # print(f'shape: {data.shape}\nNull values: {data.isnull().sum()}\n')

    if inputText is not None and inputRating is not None:
        # Create a new DataFrame using the inputText and inputRating
        new_data = pd.DataFrame({'sentiment': [inputText], 'rating': [inputRating]})
        return new_data
    else:
        return data.sample(frac=0.8).reset_index(drop=True)

def preprocess_data(data):

    return data

def get_sentiment(data):
    data['sentiment'] = data['column_1'].fillna('') + ' ' + data['column_2'].fillna('') + ' ' + data['column_4'].fillna('') + ' ' + data['column_5'].fillna('')
    data['rating'] = data[['column_0', 'column_3']].max(axis=1)
    
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

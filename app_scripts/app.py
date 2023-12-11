import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

from load_preprocess_data import load_data, preprocess_data, get_sentiment, plot_ratings
from model import huggingface_autoTokenizer, huggingFace_Distilbert
from topic_modelling import get_topics, get_common_topics , plot_topic_repetitions, plot_topic_vs_ratings, plot_interactive_ratings

import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

st.title('Sentiment Analysis of Amazon Reviews')
# st.caption('This app analyzes the sentiment of Amazon reviews')
st.text(f'The data used in this app is from the {[Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)} by Jianmo Ni, UCSD')
st.caption('The model used in this app is the [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model by Hugging Face')
st.caption('The topic modelling used in this app is the [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) model by David M. Blei, Andrew Y. Ng, Michael I. Jordan')
st.caption('The topic modelling used in this app is the [Non-negative Matrix Factorization (NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) model by Daniel D. Lee, H. Sebastian Seung')
st.caption('The topic modelling used is the [Truncated Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) model by Paul G. Constantine, David F. Gleich, Paul W. Mahoney')
st.divider()

# st.write('The interactive plots used in this app is created using [Plotly](https://plotly.com/python/) library')

# st slider and input text side by side
inputText, inputRatings = st.columns(2)
with inputText:
    inputText = st.text_input('Enter a review')
with inputRatings:
    inputRatings = st.slider('Enter a rating', 1, 5)


# inputText, inputRatings = st.text_input('Enter a review'), st.slider('Enter a rating', 1, 5)
# if inputText and inputRatings:
#     st.text(f'The review you entered is "{inputText}" and the rating you entered is {inputRatings}')

def get_data(inputText=None, inputRatings=None):
    data1 = load_data()
    data1 = get_sentiment(data1)
    # st.write(f'\n Data1',data1.shape, data1.sample(frac=.25))

    data2 = load_data(inputText=inputText, inputRating=inputRatings)
    # st.write(f'\n Data2',data2.shape, data2.sample(frac=.25))

    # Join data1 and data2
    data = pd.concat([data1, data2])
    # st.write(f'Data (Confirm the ratings)... \n',data.sample(frac=.25))

    data = huggingface_autoTokenizer(data) # This is less accurate that the DistilBERT model
    data = huggingFace_Distilbert(data)
    
    # st.write(f'Got data of shape: {data.shape}')
    return data1

if st.button('Analyze'):
    start = datetime.datetime.now()
    data = get_data(inputText=inputText, inputRatings=inputRatings)
    data = get_topics(data)
    data = get_common_topics(data)

    st.write(f'sample data ...\n {data.shape}',data.sample(frac=.25))

    st.pyplot(plot_topic_repetitions(data))
    st.pyplot(plot_topic_vs_ratings(data))
    st.write(plot_interactive_ratings(data))

    end = datetime.datetime.now()
    st.markdown(f'Time taken to analyze: `{end - start}`')

    # Print the data rows with the same common_topics as the inputText
    st.write(data[data.common_topics == data.loc[data.sentiment == inputText, 'common_topics'].values[0]][['sentiment','huggingFace_Distilbert', 'huggingface_autoTokenizer', 'common_topics']])

st.markdown(
    '`Created by` [Brenda](https://github.com/lagom-QB) | \
         `Code:` [GitHub](https://github.com/lagom-QB/AmazonReviews)')
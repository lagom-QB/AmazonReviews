import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from load_preprocess_data import load_data, preprocess_data, get_sentiment, plot_ratings
from model import huggingface_autoTokenizer, huggingFace_Distilbert
from topic_modelling import get_topics, get_common_topics , plot_topic_repetitions, plot_topic_vs_ratings, plot_interactive_ratings

import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

st.title('Sentiment Analysis of Amazon Reviews')
st.markdown('`This app analyzes the sentiment of Amazon reviews`')
st.markdown('`The data used in this app is from the` [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) `by Jianmo Ni, UCSD`')
st.markdown('`The model used in this app is the `[DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)` model by Hugging Face`\n`The model used in this app is the `[AutoTokenizer](https://huggingface.co/transformers/model_doc/auto.html)` model by Hugging Face`')
st.markdown('`The topic modelling used in this app is the `[Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)` model by David M. Blei, Andrew Y. Ng, Michael I. Jordan`')
st.markdown('`The topic modelling used in this app is the `[Non-negative Matrix Factorization (NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)` model by Daniel D. Lee, H. Sebastian Seung`')
st.markdown('`The topic modelling used is the `[Truncated Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition)` model by Paul G. Constantine, David F. Gleich, Paul W. Mahoney`')
st.divider()

# st.write('The interactive plots used in this app is created using [Plotly](https://plotly.com/python/) library')

inputText, inputRatings = st.text_input('Enter a review'), st.slider('Enter a rating', 1, 5)
if inputText and inputRatings:
    st.write(f'The review you entered is "{inputText}" and the rating you entered is {inputRatings}')

# @st.cache_data
def get_data(inputText=None, inputRatings=None):
    data1 = load_data()
    data1 = get_sentiment(data1.sample(frac=.8).reset_index(drop=True))
    # print(data1.shape)

    data2 = load_data(inputText=inputText, inputRating=inputRatings)
    # print(data2.shape)

    # Join data1 and data2
    data = pd.concat([data1, data2])

    print(data.shape)

    # data = get_sentiment(data)

    data = huggingface_autoTokenizer(data)
    data = huggingFace_Distilbert(data)
    
    return data

if st.button('Analyze'):
    data = get_data(inputText=inputText, inputRatings=inputRatings)
    data = get_topics(data)
    data = get_common_topics(data)

    plot_topic_repetitions(data)
    plot_topic_vs_ratings(data)
    plot_interactive_ratings(data)

    st.write(data.sample(4))

st.markdown(
    '`Created by` [Brenda](https://github.com/lagom-QB) | \
         `Code:` [GitHub](https://github.com/lagom-QB/Data-Engineering)')
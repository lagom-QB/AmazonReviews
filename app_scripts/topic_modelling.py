
import fire
import seaborn as sns

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter

import plotly.graph_objects as go


def get_topics(data, n_components=3):
    vectorizer = CountVectorizer(stop_words='english')
    word_doc = vectorizer.fit_transform(data['sentiment'])

    for index, row in data.iterrows():
        word_doc = vectorizer.fit_transform([row['sentiment']])

        lda_model = LatentDirichletAllocation(random_state=42)
        lda_model.fit(word_doc)

        nmf_model = NMF(random_state=42)
        nmf_model.fit(word_doc)

        svd_model = TruncatedSVD(random_state=42)
        svd_model.fit(word_doc)

        feature_names = vectorizer.get_feature_names_out()

        data.at[index, 'top_words_lda'] = ', '.join([feature_names[i] for i in lda_model.components_[0].argsort()[:-n_components - 1:-1]])
        data.at[index, 'top_words_nmf'] = ', '.join([feature_names[i] for i in nmf_model.components_[0].argsort()[:-n_components - 1:-1]])
        data.at[index, 'top_words_svd'] = ', '.join([feature_names[i] for i in svd_model.components_[0].argsort()[:-n_components - 1:-1]])

    return data

def get_common_topics(data):
    # Get the most common word in nmf, lda, svd
    for idx, row in data.iterrows():
        nmf = row['top_words_nmf'].split(', ')
        lda = row['top_words_lda'].split(', ')
        svd = row['top_words_svd'].split(', ')
        
        # Count the occurrence of each word
        word_counts = Counter(nmf + lda)
        
        # Get the most common word
        most_common_word = word_counts.most_common(1)[0][0]
        
        data.at[idx, 'common_topics'] = most_common_word

    return data

def plot_topic_repetitions(data):
    sns.set_theme(style='whitegrid')

    value_counts = data['common_topics'].value_counts()
    # value_counts to dataframe with columns 'common_topics' and 'occurrences'
    value_counts = value_counts.reset_index()
    value_counts.columns = ['common_topics', 'occurrences']

    if value_counts.occurrences.max() > 300:
        thresh_value = value_counts.occurrences.max() * 0.1
    else:
        thresh_value = value_counts.occurrences.max() * 0.01
        
    common_topics = data[data['common_topics'].isin(value_counts[value_counts['occurrences'] > thresh_value]['common_topics'])]

    topic_counts = common_topics['common_topics'].value_counts().reset_index()
    topic_counts.columns = ['common_topics', 'occurrences']

    plt.figure(figsize=(10, 16))
    sns.scatterplot(data=topic_counts,
                    x='occurrences',
                    y='common_topics',
                    size='occurrences',
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=1)

    plt.title('Common Topics in Reviews', fontsize=20)
    plt.xlabel('Number of Occurrences', fontsize=14)
    plt.ylabel('Common Topics', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()
    return plt.gcf()

def plot_topic_vs_ratings(data):
    value_counts = data.common_topics.value_counts()
    thresh_value = pd.Series(index=value_counts.index)  # Initialize thresh_value with the same index as value_counts
    thresh_value[value_counts > 300] = value_counts * 0.1
    thresh_value[value_counts <= 300] = value_counts * 0.01

    max_ratings_by_topic = data[data.common_topics.isin(value_counts[thresh_value > 0].index)].groupby('common_topics')['rating'].mean().reset_index()  # Use thresh_value > 0 to filter the index

    display(max_ratings_by_topic)

    plt.figure(figsize=(8, 20))
    sns.scatterplot(x='rating',
                    y='common_topics',
                    palette='Set1',
                    alpha=0.7,
                    linewidth=0.2,
                    legend=False,
                    data=max_ratings_by_topic)

    plt.title('Common Topics and Ratings', fontsize=18)
    plt.xlabel('Ratings', fontsize=14)
    plt.ylabel('Common Topics', fontsize=14)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim(0, 6)
    plt.gca().set_yticklabels([''] + list(max_ratings_by_topic['common_topics']))

    plt.show()
    return plt.gcf()

def plot_interactive_ratings(data):
    max_ratings_by_topic = data[data.common_topics.isin(data.common_topics.value_counts()[data.common_topics.value_counts() > 10].index)].groupby('common_topics')['rating'].mean().reset_index()

    fig = go.Figure(data=go.Scatter(
        x=max_ratings_by_topic['rating'],
        y=max_ratings_by_topic['common_topics'],
        mode='markers',
        marker=dict(
            color=max_ratings_by_topic['rating'],
            size=10
        )
    ))

    fig.update_layout(
        title='Common Topics and Average Ratings',
        xaxis_title='Ratings',
        yaxis_title='Common Topics',
        font=dict(size=10),
        height=1000,
    )

    fig.update_yaxes(automargin=True)

    fig.show()
    return fig

if __name__ == '__main__':
    fire.Fire(get_topics)
    fire.Fire(get_common_topics)
    fire.Fire(plot_topic_repetitions)
    fire.Fire(plot_topic_vs_ratings)
    fire.Fire(plot_interactive_ratings)
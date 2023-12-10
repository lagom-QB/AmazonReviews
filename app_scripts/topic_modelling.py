
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
    sns.set_theme(style='darkgrid')

    common_topics = data[data.common_topics.isin(data.common_topics.value_counts()[data.common_topics.value_counts() > 100].index)]

    # Plot the count of each common topic
    size_dict = common_topics['common_topics'].value_counts().to_dict()

    # Create a DataFrame for the value counts
    topic_counts = pd.DataFrame(list(size_dict.items()), columns=['common_topics', 'occurrences'])

    # Adjust figure size to avoid a value error
    plt.figure(figsize=(10, 18))

    # Create the bubble plot
    sns.barplot(data=topic_counts,
                    y='common_topics',  # Plot common_topics on the x-axis
                    x='occurrences',  # Plot occurrences on the y-axis
                    alpha=0.7,
                    edgecolor='white',
                    linewidth=.2)

    plt.title('Common Topics in Reviews', fontsize=20) # Add a title
    plt.xlabel('Number of Occurrences', fontsize=14) # Label the x-axis
    plt.ylabel('Common Topics', fontsize=14) # Label the y-axis
    plt.xticks(fontsize=8) # Increase the font size of the x-axis tick labels
    plt.yticks(fontsize=8) # Increase the font size of the y-axis tick labels
    
    plt.show()
    return plt.gcf()

def plot_topic_vs_ratings(data):
    # Plot the count of each common topic against the rating

    # Calculate the maximum rating from the dataset
    max_ratings_by_topic = data[data.common_topics.isin(data.common_topics.value_counts()[data.common_topics.value_counts() > 20].index)].groupby('common_topics')['rating'].mean().reset_index()

    # display(max_ratings_by_topic)

    # Adjust figure size
    plt.figure(figsize=(8, 20))

    # Create the bubble plot
    sns.scatterplot(x='rating',  # Placeholder x-value, will be overridden by hue
                    y='common_topics',  # Plot common_topics on the y-axis
                    palette='Set1',
                    alpha=0.7,
                    linewidth=0.2,
                    legend=False,
                    data=max_ratings_by_topic)  # Use the calculated maximum ratings by topic

    plt.title('Common Topics and Ratings', fontsize=18) # Add a title

    plt.xlabel('Ratings', fontsize=14) # Label the x-axis
    plt.ylabel('Common Topics', fontsize=14) # Label the y-axis

    plt.xticks(fontsize=8) # Increase the font size of the x-axis tick labels
    plt.yticks(fontsize=8) # Increase the font size of the y-axis tick labels

    plt.xlim(0, 6) # Set the x-axis limits
    # Add spacing between the y-axis tick labels
    plt.gca().set_yticklabels([''] + list(max_ratings_by_topic['common_topics']))

    plt.show()
    return plt.gcf()

def plot_interactive_ratings(data):
    max_ratings_by_topic = data[data.common_topics.isin(data.common_topics.value_counts()[data.common_topics.value_counts() > 10].index)].groupby('common_topics')['rating'].mean().reset_index()
    
    # Create a scatter plot
    fig = go.Figure(data=go.Scatter(
        x=max_ratings_by_topic['rating'],  # Use the ratings as x-values
        y=max_ratings_by_topic['common_topics'],  # Use the common_topics as y-values
        mode='markers',  # Set the mode to markers for a scatter plot
        marker=dict(
            color=max_ratings_by_topic['rating'],  # Color the markers based on the ratings
            # colorscale='turbo',  # Choose a color scale
            # colorbar=dict(title='Ratings'),  # Add a color bar with a title
            size=10  # Set the marker size
        )
    ))
    
    # Customize the layout
    fig.update_layout(
        title='Common Topics and Average Ratings',  # Add a title
        xaxis_title='Ratings',  # Label the x-axis
        yaxis_title='Common Topics',  # Label the y-axis
        font=dict(size=10),  # Set the font size
        height=1000,  # Set the height of the plot
    )

    fig.update_yaxes(automargin=True)
    
    # Show the interactive scatter plot
    fig.show()
    return fig

if __name__ == '__main__':
    fire.Fire(get_topics)
    fire.Fire(get_common_topics)
    fire.Fire(plot_topic_repetitions)
    fire.Fire(plot_topic_vs_ratings)
    fire.Fire(plot_interactive_ratings)
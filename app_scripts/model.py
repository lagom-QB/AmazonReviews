import seaborn as sns

from transformers import pipeline, AutoTokenizer # basic, autoTokinizer, DistilBertForSequenceClassifier, 

import torch
import fire

import concurrent.futures
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

"""
  Trying out various models to evaluate their efficiency
    - HuggingFace transformers AutoTokenizer
    - HuggingFace transformers distilbert-base-uncased-finetuned-amazon-reviews 
"""

def huggingface_autoTokenizer(data, batch_size=32):

  classifier = pipeline('text-classification', model='RashidNLP/Amazon-Deberta-Base-Sentiment')
  batched_outputs = []
  num_samples = len(data)

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = []

    for start in range(0, num_samples, batch_size): # iterate over the data in batches of 32
      end = min(start + batch_size, num_samples) # get the end of the batch
      batch = data['sentiment'].iloc[start:end].tolist() # get the batch
      futures.append(executor.submit(classifier, batch)) # submit the batch to the classifier

    for future in concurrent.futures.as_completed(futures): # iterate over the futures
      batch_outputs = future.result() # get the results of the future
      batched_outputs.extend(batch_outputs) # extend the batched outputs with the results

  data['huggingface_autoTokenizer'] = [output['label'] for output in batched_outputs] # add the results to the dataframe

  return data

def huggingFace_Distilbert(data):
    
    classifier = pipeline("text-classification", model="amir7d0/distilbert-base-uncased-finetuned-amazon-reviews")
    batched_outputs = []
    num_samples = len(data)

    for start in range(0, num_samples, 32):
        end = min(start + 32, num_samples)
        
        batch = data['sentiment'].iloc[start:end].tolist()
        batched_outputs.extend(classifier(batch))

    sentiment_mapping = {'1 star': "very negative", 
                                         '2 stars': "negative", 
                                         '3 stars': "neutral", 
                                         '4 stars': "positive", 
                                         '5 stars': "very positive"}
    # data['huggingFace_Distilbert'] = [sentiment_mapping[output['label']] for output in batched_outputs]
    data['huggingFace_Distilbert'] = [output['label'] for output in batched_outputs]
    
    return data

if __name__ == '__main__':
    fire.Fire(huggingface_autoTokenizer)
    fire.Fire(huggingFace_Distilbert)
# Amazon Reviews Analysis

## About

This project involves analysing and understanding the emotion expressed in text data; determining the overall sentiment of a piece of text from [amazon reviews](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view?usp=drive_link&resourcekey=0-Rp0ynafmZGZ5MflGmvwLGg).

Understanding the data, I realized there were 6 columns; 2 were ratings while the rest were split sentiments. Because of this, I ended up merging the similar columns so as to have a single sentence as a sentiment.

Regardless, an important question I asked myself was why is this project relevant? How does it help in the real-world? What would it show potential employers or employees or people who keep doing a data science project on Amazon Reviews.  

- It helps understand customer insights; Shows what customers are buying on amazon, what they are interested in buying. Using topic modelling, I extract what the reviews are mainly about. This could help potential e-commerce sites gain an insight into what to offer customers. Unfortunately, the data I am currently using doesn't offer location data so I can't filter to get what locations are buying what commodities but that would have been an interesting addition.
- The ability to analyze such a huge amount of data and extract meaningful insights from it demonstrates strong analytical skills and a thirst for actionable insights.
- Sentiment analysis and predictive modelling using this dataset though without using NLTK, shows an understanding of text analysis and the concept of pipelining since the models I trained my data on were from huggingface.

---

The project steps involve :

- Data Collection
- Preprocessing
- Feature extraction
- Model
  - Training
  - Evaluation
- Deployment

## Challenges

- Due to the large amount of the data, it was difficult to process and train the data.  
  - To overcome this, I implemented concurrency using batches after training on a quarter sample of the total data. Here, concurrency handles the complexity of multithreading and multiprocessing and enables asynchronous execution and results.

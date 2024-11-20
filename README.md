# Ironhack Bootcamp Project 3: Natural Language Processing

Build a classifier to diffecienciate real news from fake news using a kaggle dataset of labeled news.

## Installation
Use **requirements.txt** to install the required packages to run the notebooks. It is advised to use a virtual environment.
```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Notebook description

- **data_exploration.ipynb**: This notebook loads and explore the dataset
- **model_training_1.ipynb**: In this notebook we try different ML classifiers and clustering models using Bag of Word and Tf-idf encodings
- **model_training_2.ipynb**: In this notebook we are using ngrams when encoding words using BoW or Tfidf, to try to retain some context for a better classification
- **model_training_3.ipynb**: In this notebook we are exploring the impact of using word embeddings on training the previous notebooks models
- **model_training_4.ipynb**: In this notebook we are exploring the TSME clustering methods
- **model_training_5.ipynb**: In this notebook we are exploring Neural Networks
- **model_training_6_BEST_MODEL.ipynb**: This is our final delivery notebook combining preprocessing, training our best model and predicting on the test dataset

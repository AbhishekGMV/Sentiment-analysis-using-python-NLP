# Sentiment-analysis-using-python-NLP
Sentiment analysis on imdb movie dataset of over 40k reviews, using ML and NLP in python 

## Movie Reviews - Sentiment Analysis
`Python 3.7` classification of tweets (positive or negative) using `NLTK-3` and `sklearn`.

An analysis of the `twitter` data set included in the `nltk` corpus.

***
## What is in this repo


- [x] An implementation of `nltk.SVM` trained against **40000 tweets**. Implemented in `SentimentAnalysis.py`.
- [x] Using `sklearn`
  - [x] **SVM**
    - [x] `SVC`:
***

### Accuracy achieved


| **Classifier**                 | **Accuracy achieved** |
|---------------------------------|-----------------------|
| `SVC`                           | _89.0%_               |


***

## Requirements


The simplest way(and the suggested way) would be to install the the required packages and the dependencies by using either [anaconda](https://www.continuum.io/downloads) or [miniconda](http://conda.pydata.org/miniconda.html)

After that you can do

```sh
$ conda update conda
$ conda install scikit-learn nltk
```

***

#### Downloading the dataset


The dataset used in this package is bundled along with the `nltk` package.

Run your python interpreter

```python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('movie_reviews') 
```

**NOTE**: You can check system specific installation instructions from the official [`nltk` website](http://www.nltk.org/data.html)

Check if everything is good till now by running your interpreter again and importing these

```python
>>> import nltk
>>> from nltk.corpus import stopwords, movie_reviews
>>> import sklearn
>>> 
````

If these imports work for you. Then you are good to go!

***

## Running the project

1. Clone the repo 

```sh
$ git clone https://github.com/AbhishekGMV/Sentiment-analysis-using-python-NLP
$ cd Sentiment-analysis-using-python-NLP
```

2. Running the code 
    ```
    Preffered to run in a jupyter notebook.
    Place the code in a cell and execute it(Shift+Ret).
    ```

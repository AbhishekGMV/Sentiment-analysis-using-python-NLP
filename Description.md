### About the project 
The trending `Sentiment analysis` project is a type of data mining that measures the inclination of people's opinions through natural language processing (NLP),
computational linguistics and text analysis, which are used to extract and analyze subjective information from the Web(Twitter)
### How it works
- Program reads two csv files `Train.csv` and `Test.csv` for train and test data respectively.
- The function `ProcessText` removes any html tags used in texts and punctuations for more accuracy in training the data and making predictions.
- `Bag of words` transormation and `Term Frequency — Inverse Document Frequency` are applied to cleaned data.
- `DoGridSearchCV` function carries out Grid search on the tf-idf data and returns best parameters for SVC pipeline model.
- Training and testing data is fit to the pipeline model and predictions are made.  

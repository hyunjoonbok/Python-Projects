# Python Portfolio
Portfolio of data science projects from either original work or revised for a study and learning purpose. Portfolio in this repo is presented in the form of iPython Notebooks and .py files.

For a more visually pleasant experience for browsing the portfolio, check out hjbok.github.io

For a detailed code example and images, please refer to readme file in each folder under framework names.

*Note: Data used in the projects is for learning and demo purposes only*

<hr>

## Motivation

This repository was origianlly to have a record of project progress and my own learning process, but I found that it would be helpful to who wants to improve data-science skills to next-level, as it contains a numerious real-life data science example and notebooks created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/) and codes borrowed from authors who produced state-of-the-art results.

As Python is nowadays a go-to for Data-Science. I have managed to use the best out of Python to use its full functionality for not only simple EDA, but building a complex ML/DL models.

Below examples include the intense usage of industry-hot frameworks (i.e. Pytorch, Fast.ai, H2O, Grandient Boosting, etc) to produce ready-to-use results.

<hr>

## Table of contents
* [ALL Portfolio](#Projects)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)
* [Contact](#contact)

<hr>

# Projects

<hr>

- ## Machine Learning

   - ### [Tabular data / Collaborative filtering](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20Neural%20Net%20Tabular%20data.ipynb): 
   <p>
   Finding a customer who's income level. Simple ML Classification problem tackled with Fast.ai API. Executable to almost all types of tabular data to naively achieve a good baseline model in a few lines of code. Also, collaborative filtering is when you're tasked to predict how much a user is going to like a certain item.  Here I looked at "MovieLens" dataset to predict the rating a user would give a particular movie (from 0 to 5) 
	</p>
   December 10, 2019

   - ### [(Kaggle) M5_Forecasting](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20M5_Forecasting.ipynb): 
   <p>
   From Walmart sales data, forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. Pre-process (Feature Enginenering / Hyperparameter Optimization) given data and used LGB/XGB ensemble to generate a final submission. From [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/overview) in Kaggle. 
	</p>
   December 10, 2019	


   - [(Kaggle) NCAAW® 2020 ML Competition](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20NCAAW20.ipynb): To forecast the outcomes of March-Madness during rest of 2020's NCAAW games. Covers all team-by-team season games results data. Pre-processing of tabular data and ensemble of LGB/XGB generates a final submission. From [Google Cloud & NCAA® ML Competition 2020-NCAAW](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/overview) in Kaggle.  *Update: this competition was cancelled in Mar.2020 due to the COVID-19.*

   - [(Kaggle) NY Taxi Trip Duration](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20NCAAW20.ipynb): To Forecast total ridetime of taxi trips in New York City. Covers both Fast.ai and LGB version of solving the problem. From [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) in Kaggle. 

   _Library / Tools: Keras, Tensorflow, fast.ai, pandas, numpy, xgboost, lightgbm, scikit-learn, optuna, Seaborn, Matplotlib

<hr>

- ### Deeplearning
   - [Image Restoration_and_Enhancement using Generative Adversarial Network(GANs)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20%5BNew%5D%20GAN%20-%20Image%20Restoration_and_Enhancement.ipynb): Use Fast.ai framework to load image data, create generator/discriminator from images. Then create a model with a custom GAN loss function. Check error and improve on test image sets.
   
   - [DCGAN - Generate_Fake_Images](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20DCGAN%20-%20Generate_new_images.ipynb): Based on a set of celebrity images, we are generating a new set of fake images. Then compare Real Images vs. Fake Images create generator/discriminator from images. Used Pytorch to load image / create Generator/Discriminator and training loop. 
   
   - [MNIST CNN, Skip-connection (U-NET)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20MNIST%20CNN%2C%20Skip-connection%20(U-NET).ipynb): Use Fast.ai framework that's built on top of pytorch, to build a simple MNIST CNN model. Use Skip-connection to build a simpel conv-nn, which achieve a state-of-the-art result (99.6% accuracy on test-set).  
   
   - [(Kaggle) Handwritten_Image_Classification (Grapheme language)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/%5BKaggle%5D%20(Fast.ai)%20Handwritten_Image_Classification%20(Grapheme%20language).ipynb): Use Fast.ai to build a CNN model to classify a image of a langauge. From [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19) in Kaggle. Includes Load image / Genearte custom loss function / Train & Test data using Fast.ai.
   
   - [Simple CNN data Augmentation](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20Simple%20CNN%20data%20Augmentation.ipynb): Image-Augmentation on CNN model is one of the most important feature engineering steps. Here I looked at how image tranformation can be done with a built-in. Wider range of selection are availalbe in [fast.ai-vision-transform](https://docs.fast.ai/vision.transform.html) except the ones shown.
   *Things to add*: How ["Albumentation"](https://github.com/albumentations-team/albumentations) library can be used within Fast.ai framework.   
   
   - [[Kaggle] MNIST Digit Recognizer](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/%5BKaggle%5D%20(Fast.ai)%20MNIST%20Digit%20Recognizer%20Kaggle.ipynb): Kaggle version of MNIST. Use Fast.ai and transfer learning to solve. 

    _Library / Tools: Pytorch, cv2, Keras, fast.ai, pandas, numpy, Pandas, Matplotlib

<hr>

- ### Time Series
  - [(Kaggle) Sales Prediction on store items](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20TimeSeries%20-%20Sales%20Prediction.ipynb): Using Fast.ai to expand a tabular data to utilize many of columns in order to predict sales on stroes based on different situations like promotion, seaons, holidays, etc. Insights are from [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
  
  - [(Kaggle) M5_Forecasting](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20M5_Forecasting.ipynb): From Walmart sales data, forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. Pre-process (Feature Enginenering / Hyperparameter Optimization) given data and used LGB/XGB ensemble to generate a final submission. From [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/overview) in Kaggle.
  
  - [Titanic Dataset - Exploratory Analysis](https://github.com/sajal2692/data-science-portfolio/blob/master/Titanic%20Dataset%20-%20Exploratory%20Analysis.ipynb): Exploratory Analysis of the passengers onboard RMS Titanic using Pandas and Seaborn visualisations.
  
  - [Stock Market Analysis for Tech Stocks](https://github.com/sajal2692/data-science-portfolio/blob/master/Stock%20Market%20Analysis%20for%20Tech%20Stocks.ipynb): Analysis of technology stocks including change in price over time, daily returns, and stock behaviour prediction.

   _Library / Tools: Keras, Tensorflow, fast.ai, pandas, numpy, xgboost, lightgbm, scikit-learn, optuna, Seaborn, Matplotlib

<hr>

- ### NLP/TextClassification

   - [Text Classification_final (Language Model)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20Neural%20Net%20Tabular%20data.ipynb): 2-way polarity (positive, negative) classification system for tweets. Using Fast.ai framework to fine-tune a language model and build a classification model with close to 80% accuracy. 
   
   - [BERT-base: classify twitter sentiment](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/(Pytorch)%20(2)%20BERT-base%20to%20Classify%20Twitter%20NLP.ipynb): Used Pytorch to encode/tokenize/train/evaluate model. The most simple version
   - [BERT-large: classify twitter sentiment](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/(Pytorch)%20(3)%20BERT-large%20to%20Classify%20Twitter%20NLP.ipynb): Using large BERT (takes longer)
	
   - [Google-play App Review Sentiment Analysis with BERT](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/(Pytorch)%20Sentiment%20Analysis%20with%20Transformer%20BERT.ipynb): 3-way polarity (positive, neutral, negative) sentiment analysis system for Google-Play App reviews. Use Pytorch to get review in JSON, data-preprocess, Create pytorch dataloader , train/evaluate the model. Evaluate the errors and testing on the raw text data in the end.
   
   - [(Kaggle) Handwritten_Image_Classification (Grapheme language)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/%5BKaggle%5D%20(Pytorch)%20Handwritten_Image_Classification%20(Grapheme%20language).ipynb): Pytorch version of builing a CNN model to classify a image of a langauge. Complete model building from loading/defining/transforming data to create and train model. From [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19) in Kaggle. 

   _Library / Tools: Pytorch, transformers, fast.ai, tqdm, pandas, numpy, pygments, google_play_scraper, albumentations, joblib, xgboost, lightgbm, scikit-learn, optuna, Seaborn, Matplotlib

<hr>

- ### Micellenous
   - [ElasticSearch connections with Python](https://github.com/hyunjoonbok/Python-Projects/blob/master/ATG_work/%5BATG%5D%20ElasticSearch%20connections%20with%20Python-v2.ipynb): Use of Python language to pull data directly from ELK stack. Origianlly came in to JSON format, convert it to Dataframe and do simple EDA / Visualization.
   
   _Library / Tools: pandas, numpy, elasticsearch, datetime
   
<hr>

## Technologies
* *Fast.ai*
* *Pytorch*
* *Tensorflow*
* *Keras*
* *CV2*
* *tqdm*
* *pandas*
* *numpy*
* *albumentations*
* *datetime*
* *xgboost*
* *lightgbm*
* *scikit-learn*
* *optuna*
* *Seaborn*
* *Matplotlib*
* *elasticsearch*
* *And More...*

<hr>

## Setup
#### - reqiurement
```
pip install -r requirements.txt
```

## Code Examples
Show examples of usage: `put-your-code-here`

<hr>

## TO-DOs
List of features ready and TODOs for future development
* Tableau Public - Add visulization using work data : _in progress_
* Python Dash for intractive wep-app : _in progress_
* Data cleaning .ipynbs : _in progress_


## Contact
Created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/) - feel free to contact me!

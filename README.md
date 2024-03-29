# Python Portfolio
Portfolio of data science projects from either original work or revised for a learning purpose. 

Portfolio in this repo is presented in the form of Jupyter Notebooks or .py files.

For a detailed code examples and breif explanation for each, please read through readme.md file below.

*Note: Data used in the projects is for demo purposes only*

<hr>

## Motivation

This repository was originally to have a record of personal project progress and my own learning process, but I found that potential data professionals would beneift from the collection of materials, as it contains a numerious real-life data science example and notebooks created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/).

As Python is nowadays a go-to for Data-Science. I have managed to use the best out of Python to use its full functionality for not only simple EDA, but building a complex ML/DL models.

Below examples include the intense usage of industry-hot frameworks (i.e. Pytorch, Fast.ai, H2O, Grandient Boosting, etc) to produce ready-to-use results.

<hr>

## Table of contents
* [ALL Portfolio](#Projects)
* [Technologies](#technologies)
* [Reference](#Reference)
* [Contact](#contact)

<hr>

# Projects

   #### [Measuring Customer Lifetime Value (LTV)](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Cohort_Basket_Analysis/Measuring%20Customer%20Lifetime%20Value%20(LTV)%20in%20Python.ipynb): 
   <p>
   In this notebbok, we measure the customer LTV (Lifetime Value) for any custom timeframe we want. We form an example using a real-world marketing dataset provided by Kaggle. We learn the concept of LTV, and how to preprocess and visualize the data to get the basic findings. And finally build a model that predicting 3-Month CLV for each customer group features.
	</p>


   #### Cohort Analysis
   [Cohort Analysis - Customer Retention (1)](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Cohort_Basket_Analysis/%5BBasic%5D%20Cohort%20Analysis%20-%20Customer%20Retention%20(1).ipynb) \
   [Cohort Analysis - Customer Retention (2)](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Cohort_Basket_Analysis/%5BAdvanced%5D%20Cohort%20Analysis%20-%20Customer%20Retention%20(2).ipynb) \
   [Cohort Analysis - Customer Segmentation (1)](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Cohort_Basket_Analysis/%5BAdvanced%5D%20Cohort%20Analysis%20-%20Customer%20Segmentation%20(1).ipynb) \
   [Cohort Analysis - Customer Segmentation (2)](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Cohort_Basket_Analysis/%5BAdvanced%5D%20Cohort%20Analysis%20-%20Customer%20Segmentation%20(2).ipynb)    
   <p>
   Suppose that we have a company that selling some of the product, and you want to know how well does the selling performance of the product. We have the data that can we analyze, but what kind of analysis that we can do? We can segment customers based on their buying behavior on the market. These notebook introduces several ways to segment the users and better understand their retention, using the K-Means algorithm in Python. Using a industry marketing data, We create cohorts to understand metrics like customer retention rates, the average quantity purchased, the average price, etc. The notebook covers a full analysis (preprocessing + visualization + interpretation) to do customer segmentation step-by-step.
	</p>
   

   #### [Marketing Channel Attribution with Markov Chains](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Cohort_Basket_Analysis/Marketing%20Channel%20Attribution%20with%20Markov%20Chains%20.ipynb): 
   <p>
   In businses marketing scenarios, it is quite impoerntat to track the conversion from the user base through the advertisement money spent. But it is more valuable to know **how** each of those conversion are made, so that the further actions are taken in on-going basis (i.e. budget adjustment). The notebook introduces a concept of 'Markov Chains' to understand the attributions in marketing. By using these transition probabilities, we can identify the statistical impact a single channel has on our total conversions.
	</p>


   #### [Market Basket Analysis 101 with Real Example - Association rules, Lift, Confidence, Support](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Cohort_Basket_Analysis/Market%20Basket%20Analysis%20101%20with%20Real%20Example%20-%20Association%20rules%2C%20Lift%2C%20Confidence%2C%20Support.ipynb): 
   <p>
   The notebook has the implementation of Basket anayiss in real-world data using Python. It goes over the concept, along with key terms and metrics aimed at giving a sense of what “association” in a rule means and some ways to quantify the strength of this association. The entire data mining process (preprocessing + visualization + interpretation) are clearly explained.
	</p>


   #### [Simple Text Mining concept and Practice from scratch](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/TextModel/Simple%20Text%20Mining%20concept%20and%20practice%20from%20scratch.ipynb): 
   <p>
   The purpose of the notebook is to get the very basic understading of the methods behind the common NLP project. It introduces 3 different approches that could be taken when performing a text-mining, from it's concept and actual implementation of codes: 1) Frequency of appearance of two words, 2) Statistical method of extracting connection, 3) Word2vec (DL). It also narrates 2 prerequisite steps to be taken before performing text-mining: 1) Select the target word 2) Choose the context: choose what is the sentence about
	</p>


   #### [Time Series Forecasting With Prophet in Python - WIP](https://github.com/hyunjoonbok/Python-Projects/blob/master/vanilla/Time%20Series%20Forecasting%20With%20Prophet.ipynb): 
   <p>
	</p>


   #### Machine Learning / Deep Learning with H2O
   [Complete guide to Machine Learning with H2O (AutoML)
](https://github.com/hyunjoonbok/Python-Projects/blob/master/H2O/Complete%20guide%20to%20Machine%20Learning%20with%20H2O%20(AutoML).ipynb) \
   [Machine Learning Regression problem with H2O (XGBoost & Deeplearning)](https://github.com/hyunjoonbok/Python-Projects/blob/master/H2O/Machine%20Learning%20Regression%20problem%20with%20H2O%20(XGBoost%20%26%20Deeplearning).ipynb) 
   <p>
   
In this notebook, we will use the subset of the Freddie Mac Single-Family dataset to try to predict the interest rate for a loan using H2O's XGBoost and Deep Learning models. We will explore how to use these models for a regression problem, and we will also demonstrate how to use H2O's grid search to tune the hyper-parameters of both models. We're going to use machine learning with H2O-3 to predict the interest rate for each loan. To do this, we will build two regression models: an XGBoost model and a Deep Learning model that will help us find the interest rate that a loan should be assigned. Complete this tutorial to see how we achieved those results. Also, we go over H2O's AutoML solution, which is an automated algorithm for automating the machine learning workflow, which includes some light data preparation such as imputing missing data, standardization of numeric features, and one-hot encoding categorical features. It also provides automatic training, hyper-parameter optimization, model search, and selection under time, space, and resource constraints. H2O's AutoML further optimizes model performance by stacking an ensemble of models. H2O AutoML trains one stacked ensemble based on all previously trained models and another one on the best model of each family.
	</p>


   #### Explainable Machine Learning with SHAP
   [Understand Classification Model with SHAP](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/ExplainableML/%5BExplainable%20Machine%20Learning%5D%20Detailed%20Bar%20Plots%20and%20Waterfall%20Plots%20in%20SHAP.ipynb) \
   [Understand Regression Model with SHAP](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/ExplainableML/%5BExplainable%20Machine%20Learning%5D%20Understand%20Regression%20Model%20with%20SHAP%20(XGBoost).ipynb) \
   [SHAP Decision Plots in Depth](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/ExplainableML/%5BExplainable%20Machine%20Learning%5D%20SHAP%20Decision%20Plots%20in%20Depth.ipynb)
   <p>
   
SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. Here, we look at the implementation of Tree SHAP, a fast and exact algorithm to compute SHAP values for trees and ensembles of trees. We have 3 different basic examples (regression / classifcation / more in-depth graphics) that can be applied to visualizaing the model.
	</p>


   #### [Multi-Class Text Classification 1 (with PySpark and Doc2Vec)](https://github.com/hyunjoonbok/natural-language-processing/blob/master/07_Multi-Class_Text_Classification_with_PySpark_and_Doc2Vec.ipynb): 
   <p>
    In this notebook, we utilize Apache Spark's machine learning library (MLlib) with PySpark to tackle NLP problem and how to simulate Doc2Vec inside Spark envioronment. Apache Spark is a famous distributed competiting system to to scale up any data processing solutions. Spark also provides a Machine-learning powered library called 'MLlib'. We utilize Spark Machine Learning Library (Spark MLlib) to look at 3297 labeled sentences, and classify them into 5 different categories.
	</p>

   #### [Multi-class Text Classification 2 (with PySpark, MLlib, SparkSQL)](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/TextModel/Multi-class%20Text%20Classification%20Problem%20with%20PySpark%20and%20MLlib.ipynb): 
   <p>
    Apache Spark is quickly gaining steam both in the headlines and real-world adoption, mainly because of its ability to process streaming data. With so much data being processed on a daily basis, it has become essential for us to be able to stream and analyze it in real time. We use Spark Machine Learning Library (Spark MLlib) to classify crime rescription into 33 categories.
	</p>

   #### [Retail Price Recommendation model with Gradient Boosting Tree](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Others/Retail%20Price%20Recommendation%20model%20with%20Gradient%20Boosting%20Tree.ipynb): 
   <p>
    Mercari (Japan’s biggest shopping app) would like to offer pricing suggestions to sellers, but this is not easy because their sellers are enabled to put just about anything, or any bundle of things, on Mercari’s marketplace. In this machine learning project, we will build a model that automatically suggests the right product prices. Here we build a complete price recommendation model leveraging LightGBM.
	</p>


   #### [Full Pytorch Implementation of Recommender System (Collaborative Filtering)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/Full%20Pytorch%20Implementation%20of%20Recommender%20System%20(Collaborative%20Filtering).ipynb): 
   <p>
   We utilize Pytoch's embeddings layers to build a simple recommendation system. Our model will predict user ratings for specific movies.
	</p>
   

   #### [End-to-End Machine Learning Model using PySpark and MLlib](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Others/End-to-End%20Machine%20Learning%20Model%20using%20PySpark%20and%20MLlib.ipynb): 
   <p>
    We build a complete ML model (Binary Classification with Imbalanced Classes problem) leveraging Spark's computation. Full cycle of ML (EDA, feature engineering, model building) is covered. In-Memory computation and Parallel-Processing are some of the major reasons that Apache Spark has become very popular in the big data industry to deal with data products at large scale and perform faster analysis
	</p>


   #### [ML Model for predicting a Crew Size](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/Others/Ship_Crew_Size_ML_Model.ipynb): 
   <p>
    EDA-focused regression model building to predict a ship's Crew Size. CSV Dataset included in a same folder.
	</p>
 
 
   #### [Simple Text Mining concept and practice from scratch](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/TextModel/Simple%20Text%20Mining%20concept%20and%20practice%20from%20scratch.ipynb): 
   <p>
    This notebook shows 3 different approches that could be taken when performing a text-mining, from it's concept and actual implementation of codes. Text mining is an approach to find a relationship between two words in a given sentence. It could be found by using: 1) Frequency of appearance of two words 2) Statistical method of extracting connection 3) Word2vec (DL)
	</p>
 

   #### [Reuters News Text Classification in Tensorflow Keras](https://github.com/hyunjoonbok/Python-Projects/blob/master/tensorflow/Reuters%20News%20Classification%20using%20LSTM%20(long-short%20tem%20memory)%20in%20Tensorflow.ipynb): 
   <p>
    We build a text classifcation on Reuters News (available through sklearn) based on LSTM method using Tensorflow
	</p>


   #### [Text Classification with MLP (MultiLayer Perceptron) in Tensorflow Keras](https://github.com/hyunjoonbok/Python-Projects/blob/master/tensorflow/Text%20Classification%20with%20MLP%20(MultiLayer%20Perceptron)%20in%20Tensorflow%20Keras.ipynb): 
   <p>
   Simple MLP model in Tensorflow to solve the text classification problem. Here, we will use the texts_to_matrx() function in Keras to perform text-classification. 
	</p>


   #### Recommedation System - Collaborative Filtering
   [FastAI Implementation](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/RecommendationModel/Recommendation%20System%20-%20Matrix%20Factoization%20(Collaboratvie%20Filtering)%20in%20FastAI.ipynb) \
   [Surprise Library Implementation](https://github.com/hyunjoonbok/Python-Projects/blob/master/GeneralML/RecommendationModel/Recommendation%20System%20-%20Matrix%20Factoization%20(Collaboratvie%20Filtering)%20using%20Surprise%20Library.ipynb)
   <p>
   Experiment with the MovieLens 100K Data to provide movie recommendations for users based on different settings (Item-based, user-based, etc)
	</p>



   #### [TabNet in Pytorch - New Solution to Tabular ML problems](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/TabNet%20in%20Pytorch%20-%20New%20Solution%20to%20Tabular%20ML%20problems.ipynb): 
   <p>
   An introduction of TabNet, which is a neural-net based algorithm to be readily used in Tabular dataset Machine Learning problems (most common in Kaggle Competitions). A Pytorch Implementation with a Toy example (adult census income dataset and forest cover type dataset) are shown in this notebook, along with a basic architecture and workflow.
	</p>


   #### [Image Segmentation using a modified U-Net in Tensorflow](https://github.com/hyunjoonbok/Python-Projects/blob/master/tensorflow/Image%20Segmentation%20using%20a%20modified%20U-Net%20in%20Tensorflow.ipynb): 
   <p>
   An image segmentation task with Oxford-IIIT Pet Dataset to build a model that genenarte masks around the pet images and eventaully segment the image itself. Built using MobileNetV2 pretrained on ImageNet.
	</p>


   #### [Rock Paper Scissors (using MobileNetV2) in Tensorflow 2.0](https://github.com/hyunjoonbok/Python-Projects/blob/master/tensorflow/Build%20RPS%20image%20classification%20using%20Tensorflow.ipynb): 
   <p>
   CNN model using Tensorflow that recognizes Rock-Paper-Scissors. Built using MobileNetV2 pretrained on ImageNet.
	</p>


   #### [Few Shot Learning (N-shot) in Pytorch](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/Prototypical%20Networks%20for%20Few-Shot-Learning.ipynb): 
   <p>
   Pytoch implementation of N-shot learning. We look at image classification of word image in many different languages (Omniglot Dataset) to and build the model that determines which of the evaluatiion set classes the sample belongs to.
	</p>


   #### [f-AnoGAN in Pytorch](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/f-AnoGAN%20(Image%20Anomaly%20Detection)%20in%20Pytorch%20.ipynb): 
   <p>
   Concept and codes for the fast unsupervised anomaly detection with generative adversarial networks (GAN), which is widely used for real-time anomaly detection applications. Uses "DCGAN" model, which is State-of-the-Art GAN model.
	</p>


   #### [Transfer Learning in Pytorch by building CIFAR-10 model](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/Transfer%20Learning%20in%20Pytorch%20by%20building%20CIFAR-10%20model.ipynb): 
   <p>
   Transfer learning explained. Modify a few last layers to fit-in to my own dataset.
	</p>
   

   #### [Pytorch Training Loop Implementation](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/Simple%20Pytorch%20Training%20Loop%20Implementation.ipynb): 
   <p>
   A simple walkthrough of training loops and metrics used in learning in Pytorch, follow by a complete example in the last using CIFAR-10 dataset.
	</p>


   #### [Recommender System (Collaborative filtering)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/Recommender%20System%20(Collaborative%20filtering).ipynb): 
   <p>
   A complete guide to recommendation system using Collaborative Filtering: Matrix Factorization. Concepts that are used in industry are explained, and compare model/metrics and build prediction algorithm.
	</p>


   #### [Neural Transfer Using PyTorch (VGG19)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/Neural%20Transfer%20Using%20PyTorch%20(VGG19).ipynb): 
   <p>
   Style transfer in practice using Pytorch using pretrained VGG19 model.
	</p>


   #### [Pytorch Training in Pratice](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/Basic%20Pytorch%20Concepts%20in%20practice%20by%20building%20MNIST%20CNN%20model%20.ipynb): 
   <p>
   Going through a complete modeling step in Pytorch based on MNIST dataset. Can grasp a general idea of Pytorch concept.
	</p>


   #### [Tensorboard usage in Pytorch](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/(Pytorch)%20VISUALIZING%20MODELS%2C%20DATA%2C%20AND%20TRAINING%20WITH%20TENSORBOARD.ipynb): 
   <p>
   How to use Tensorboard in Jupyter notebook when training a model in Pytorch.
	</p>


   #### [Google-play App Review Sentiment Analysis with BERT](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/(Pytorch)%20Sentiment%20Analysis%20with%20Transformer%20BERT.ipynb): 
   <p>
   3-way polarity (positive, neutral, negative) sentiment analysis system for Google-Play App reviews. Use Pytorch to get review in JSON, data-preprocess, Create pytorch dataloader , train/evaluate the model. Evaluate the errors and testing on the raw text data in the end.
	</p>


   #### [Credit Card Fraud Detection using Keras (Imbalanced response)](https://github.com/hyunjoonbok/Python-Projects/blob/master/tensorflow/Credit%20Card%20Fraud%20Dectection%20using%20Keras%20(Imbalanced%20response).ipynb) 
   <p>
   Buiding a Fraud Detection model using a sample Credit Card transaction data from Kaggle. The data is highly imbalanced, so it shows how to adjust sampling to solve the problem. Then we check important metrics needed to be evalulated (fp/tp/precision/recall, etc)
   </p>
   Reference: [Kaggle CreditCard data](https://www.kaggle.com/mlg-ulb/creditcardfraud/)
   
   
   #### [(Kaggle) Handwritten_Image_Classification (Grapheme language)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/%5BKaggle%5D%20(Pytorch)%20Handwritten_Image_Classification%20(Grapheme%20language).ipynb): 
   <p>
   Pytorch version of builing a CNN model to classify a image of a langauge. Complete model building from loading/defining/transforming data to create and train model. From [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19) in Kaggle. 
	</p>
   
   
   #### [(Kaggle) M5_Forecasting](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20M5_Forecasting.ipynb): 
   <p>
   From Walmart sales data, forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. Pre-process (Feature Enginenering / Hyperparameter Optimization) given data and used LGB/XGB ensemble to generate a final submission. From [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/overview) in Kaggle. 
	</p> 


   #### [(Kaggle) NCAAW® 2020 ML Competition](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20NCAAW20.ipynb): 
   <p>
   To forecast the outcomes of March-Madness during rest of 2020's NCAAW games. Covers all team-by-team season games results data. Pre-processing of tabular data and ensemble of LGB/XGB generates a final submission. From [Google Cloud & NCAA® ML Competition 2020-NCAAW](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/overview) in Kaggle.  *Update: this competition was cancelled in Mar.2020 due to the COVID-19.*
	</p>
   

   #### [Text Classification_final (Language Model)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20Text%20Classification%20V2%20(Language%20Model).ipynb): 
   <p>
   2-way polarity (positive, negative) classification system for tweets. Using Fast.ai framework to fine-tune a language model and build a classification model with close to 80% accuracy. 
	</p>


<hr>
- ## Machine Learning
	   Library / Tools: Keras, Tensorflow, fast.ai, pandas, numpy, xgboost, lightgbm, scikit-learn, optuna, Seaborn, Matplotlib


   ### [Tabular data / Collaborative filtering](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20Neural%20Net%20Tabular%20data.ipynb): 
   <p>
   Finding a customer who's income level. Simple ML Classification problem tackled with Fast.ai API. Executable to almost all types of tabular data to naively achieve a good baseline model in a few lines of code. Also, collaborative filtering is when you're tasked to predict how much a user is going to like a certain item.  Here I looked at "MovieLens" dataset to predict the rating a user would give a particular movie (from 0 to 5) 
	</p>
   
   
    ### [(Kaggle) Handwritten_Image_Classification (Grapheme language)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/%5BKaggle%5D%20(Fast.ai)%20Handwritten_Image_Classification%20(Grapheme%20language).ipynb): 
   <p>
   Use Fast.ai to build a CNN model to classify a image of a langauge. From [Bengali.AI Handwritten Grapheme Classification](https://www.kaggle.com/c/bengaliai-cv19) in Kaggle. Includes Load image / Genearte custom loss function / Train & Test data using Fast.ai.
	</p>  
   
   
   ### [(Kaggle) NY Taxi Trip Duration](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20NY%20Taxi%20Data.ipynb):
   <p>
   To Forecast total ridetime of taxi trips in New York City. Covers both Fast.ai and LGB version of solving the problem. From [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) in Kaggle. 
	</p>
   

<hr>

- ## Deep Learning
	    Library / Tools: Pytorch, cv2, Keras, fast.ai, pandas, numpy, Pandas, Matplotlib

   ### [Image Restoration_and_Enhancement using Generative Adversarial Network(GANs)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20%5BNew%5D%20GAN%20-%20Image%20Restoration_and_Enhancement.ipynb): 
   <p>
   Use Fast.ai framework to load image data, create generator/discriminator from images. Then create a model with a custom GAN loss function. Check error and improve on test image sets.
	</p>
   
   ### [DCGAN - Generate_Fake_Images](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20DCGAN%20-%20Generate_new_images.ipynb): 
   <p>
   Based on a set of celebrity images, we are generating a new set of fake images. Then compare Real Images vs. Fake Images create generator/discriminator from images. Used Pytorch to load image / create Generator/Discriminator and training loop. 
	</p>
   
   ### [MNIST CNN, Skip-connection (U-NET)](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20MNIST%20CNN%2C%20Skip-connection%20(U-NET).ipynb): 
   <p>
   Use Fast.ai framework that's built on top of pytorch, to build a simple MNIST CNN model. Use Skip-connection to build a simpel conv-nn, which achieve a state-of-the-art result (99.6% accuracy on test-set).
	</p>
   
   
   ### [Simple CNN data Augmentation](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20Simple%20CNN%20data%20Augmentation.ipynb): 
   <p>
   Image-Augmentation on CNN model is one of the most important feature engineering steps. Here I looked at how image tranformation can be done with a built-in. Wider range of selection are availalbe in [fast.ai-vision-transform](https://docs.fast.ai/vision.transform.html) except the ones shown.
   *Things to add*: How ["Albumentation"](https://github.com/albumentations-team/albumentations) library can be used within Fast.ai framework.   
	</p>
   
   
   ### [(Kaggle) MNIST Digit Recognizer](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/%5BKaggle%5D%20(Fast.ai)%20MNIST%20Digit%20Recognizer%20Kaggle.ipynb): 
   <p>
   Kaggle version of MNIST. Use Fast.ai and transfer learning to solve. 
	</p>


<hr>

- ## Time Series
	   Library / Tools: Keras, Tensorflow, fast.ai, pandas, numpy, xgboost, lightgbm, scikit-learn, optuna, Seaborn, Matplotlib

   ### [(Kaggle) Sales Prediction on store items](https://github.com/hyunjoonbok/Python-Projects/blob/master/Fast.ai/(Fast.ai)%20TimeSeries%20-%20Sales%20Prediction.ipynb): 
   <p>
   Using Fast.ai to expand a tabular data to utilize many of columns in order to predict sales on stroes based on different situations like promotion, seaons, holidays, etc. Insights are from [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
	</p>
	
  
   ### [(Kaggle) M5_Forecasting](https://github.com/hyunjoonbok/Python-Projects/blob/master/Kaggle/%5BKaggle%5D%20M5_Forecasting.ipynb):
   <p>
   From Walmart sales data, forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. Pre-process (Feature Enginenering / Hyperparameter Optimization) given data and used LGB/XGB ensemble to generate a final submission. From [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/overview) in Kaggle.
	</p>


<hr>


- ## NLP/TextClassification
	   Library / Tools: Pytorch, transformers, fast.ai, tqdm, pandas, numpy, pygments, google_play_scraper, albumentations, joblib, xgboost, lightgbm, scikit-learn, optuna, Seaborn, Matplotlib

   
   ### [BERT-base: classify twitter sentiment](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/(Pytorch)%20(2)%20BERT-base%20to%20Classify%20Twitter%20NLP.ipynb): 
   <p>
   Used Pytorch to encode/tokenize/train/evaluate model. The most simple version
	</p>

   
   ### [BERT-large: classify twitter sentiment](https://github.com/hyunjoonbok/Python-Projects/blob/master/Pytorch/(Pytorch)%20(3)%20BERT-large%20to%20Classify%20Twitter%20NLP.ipynb): 
   <p>
   Using large BERT (takes longer)
	</p>
	

<hr>

- ## Micellenous
	   Library / Tools: pandas, numpy, elasticsearch, datetime

   ### [ElasticSearch connections with Python](https://github.com/hyunjoonbok/Python-Projects/blob/master/ATG_work/%5BATG%5D%20ElasticSearch%20connections%20with%20Python-v2.ipynb): 
   <p>
   Use of Python language to pull data directly from ELK stack. Origianlly came in to JSON format, convert it to Dataframe and do simple EDA / Visualization.
	</p>
   
   
<hr>

## Technologies
* *Fast.ai*
* *Pytorch*
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

## Reference
- [AWS Tech Blog](https://aws.amazon.com/ko/blogs/aws/)
- Deep Learning Model Implementation Zoo (Tensorflow 1 and Pytorch) [Github](https://github.com/rasbt/deeplearning-models?fbclid=IwAR15xtWohLZCyhNd8mpuFmhK-PhvqzMFsWFaxDaqXsQVqlRrj0-sIFanqvQ)

<hr>

## Contact
Created by [@hyunjoonbok](https://www.linkedin.com/in/hyunjoonbok/) - feel free to contact me!

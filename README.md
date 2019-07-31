# Machine Learning Nanodegree

Projects required for the Machine Learning Engineer Nanodegree from Udacity. The program teaches you how to build machine learning models and apply them to data sets in fields like finance, healthcare, education, and more. This is a one term program spread over 6 months. 

## Model Evaluation and Validation: Predicting Boston Housing Prices

_Project 1_: Use Scikit-Learn and regression to analyze a real dataset of housing prices in Boston. Build and optimize a model to predict the price of houses, based on their features.

I evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts and available on [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php). A model trained on this data that is seen as a good fit could then be used to make certain predictions about the monetary value of a home. 

_The model uses a Decision Tree Algorithm. The work is done on a template Jupyter Notebook provided by Udacity. The code is written in Python 3 using NumPy and Pandas._

[Link: Project1](http://htmlpreview.github.io/?https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P1.Boston_Housing.html)

## Supervised Learning: Finding Donors for CharityML

_Project 2_: Build a classification model to identify the best potential donors for a charity. Use several classification algorithms and optimize them for quality.

I employ three supervised algorithms (Support Vector Machines, K-Nearest Neighbors and Random Forest) to accurately model individuals' income using data collected from the 1994 U.S. Census, and available on [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php), in order to identify people most likely to donate to CharityML (a fictitious charity organization). From preliminary results, the best candidate algorithm is Random Forest. I further optimize this algorithm to best model the data.

_The project is written in Jupyter Notebook, under Python 2.7 and uses NumPy and Pandas._

[Link: Project2](http://htmlpreview.github.io/?https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P2.Finding_Donors.html)

## Unsupervised Learning: Creating Customer Segments

_Project 3_: Study a real dataset of customers for a company, apply several unsupervised learning techniques in order to extract valuable information from this data.

In this project, I analyze a dataset (available on  [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)), containing data on various customers' annual spending amounts (reported in monetary units) on diverse product categories. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.

First, I explore the data to determine if any product category highly correlates with another. Afterwards, I preprocess the data by scaling each feature and then identifying to which extent the features are still correlated. After removing outliers, on the cleaned data, I implement Principal Component Analysis to find which compound combination of features best describe the customers. Next, a Gaussian Mixture Model Clustering Algorithm is used to identify hidden customer segments. Finally, the found segmentation is compared to an additional labeling, and I find ways this information could assist the wholesale distributor with future service changes.

_The project is written in Jupyter Notebook, under Python 2.7 and it uses a template provided by Udacity._

[Link: Project3](http://htmlpreview.github.io/?https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P3.Customer_Segments.html)

## Deep Learning: Dog Breed Recognition Project

_Project 4_: Use Keras to build a convolutional neural network to recognize dog breeds from images. Furthermore, if an image of a human is provided, the algorithm will guess which breed of dog looks the most like that human. Improve the results by using pre-trained networks.

I work with two datasets. The dataset of dog images contains 8351 images in 133 dog categories. This set is split into a 6680 images training set, a validation set with 835 images and a test set with 836 images. The second dataset contains 13233 human images.

First, I use OpenCV implementation of Haar cascade classifier to detect human faces in images. This is achieved with a pre-trained face detector available on Github. Secondly, I use a pre-trained ResNet-50 model to detect dogs in images. The weights of this Convolutional Neural Network were pre-trained on ImageNet, a very large dataset used for image classification and other vision tasks. This is performed in Keras.
Thirdly, I create a CNN that consists of several groups of convolutional layers, batch normalization layers and maxpooling layers, the final layers are a global averge pooling layer and a dense layer. I use a RMSprop optimizer and ran the algorihm on 10 epochs to obtain a 23.44% accuracy. Comparing to a pre-trained VGG-16 model provided by Udacity, I create a CNN that identifies dog breeds and uses a pre-trained Xception model. After a run on 20 epochs this model reaches an accuracy of 85.17%. Finally, I applied the model to predict the dog breeds for several new images. I also combine these results with results of the human face detector to determine which dog breed a human resembles the most.

_This project is based on template code and guidance provided by Udacity. The work is done on a Jupyter notebook, using Python3._

[Link: Project4](http://htmlpreview.github.io/?https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P4.Dog_Breeds_Recognition.html)

## Reinforcement Learning: Teach a Quadcopter to Fly

_Project 5_: Use deep reinforcement learning to design an agent that can fly a simulated quadcopter that learns to take off, hover, and land, all by itself. Integrate these behaviors into a single end-to-end system, that can autonomously fly from point A to point B.

To accomplish this task I used the sample actor - critic model provided by Udacity based on the Deep Deterministic Policy Gradient algorithm. I also included an Ornstein–Uhlenbeck noise process and a buffer to store the experiences.

The actor neuronal net has an input layer, followed by a three groups of hidden layers. In each group of hidden layers there is a Dense layer with L2 kernel regularizer, a Batch Normalization layer and a RELu activation layer. The Dense output layer has an Uniform kernel initializer and Sigmoid activation. The critic neronal net has actions and states input layers. There are two groups of hidden layers for the state pathway. The first group contains a Dense, a Batch Normalization layer and RELu activation layer, the second group has a Dense layer and a RELu activation layer. For the action pathway there are two Dense layers with L2 regularizers and RELu activations.

_This project is based on a template Jupyter notebook and sample code provided by Udacity. The work is done in Python3, using NumPy, Pandas and Matplotlib, and also the machine learning library TensorFlow._

[Link: Project5](http://htmlpreview.github.io/?https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P5.Flying_Quadcopter.html)

## Capstone Project

_Project 6_: Given a large set of text documents, perform sentiment analysis using deep learning techniques and supervised machine learning methods. The outcome should provide a binary classification of each document, with 0
indicating a negative sentiment and 1 indicating a positive sentiment.

The data used in this project is from the Kaggle public dataset repository: Amazon Reviews: Unlocked Mobile Phones  [link](https://www.kaggle.com/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones), it consists of more than 400,000 reviews from Amazon's unlocked mobile phone category, extracted by PromptCloud in December 2016. There are 6 features: Product Name, Brand Name, Price, Rating, Reviews, Review Votes.

The main goal of my analysis is to automatically determine whether the review expresses an overall positive or negative opinion about the product. In order to achieve this goal, I employ Convolutional Neural Networks in
conjunction with advanced text processing techniques such as Tf-Idf, word embedings and the SentimentAnalyzer module from NLTK. I use a Support Vector Machine (SVM) classifier as a benchmark model. 

[Project proposal](https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P6.Capstone_Proposal.pdf): In this preliminary work, I define the problem to be solved and I investigate potential solutions and performance metrics. 

[Data preparation](http://htmlpreview.github.io/?https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P6.Data_Preparation.html): I prepare and analyze the data through descriptive statistics and visualizations, to have a better understanding of what algorithms and features are appropriate for working with it. This includes basic text features extraction and analysis.

[Data processing](http://htmlpreview.github.io/?https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P6.Data_Processing.html): A preliminary evaluation of the cleaned data is obtained using the Vader sentiment analyzer in NLTK. The cleaned data is processed using a Tf-Idf word embedding and a Word2Vec embedding. I implement the benchmark model (SVM) and the convolutional neural network (CNN).

[Project report](https://github.com/SolanaO/mlen_udacity/blob/master/mlen.P6.Capstone_Report.pdf): The report includes information on the performance of the models used, visualizations of the relevant features, and conclusions about the results.

_The work is completed in a couple of Jupyter notebooks, using Python 3 and the following dependencies: Tensorflow, Pandas, Numpy, Matplotlib, Scikit-learnand Nltk._






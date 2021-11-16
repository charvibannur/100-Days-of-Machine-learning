# 100-Days-of-Machine-learning
100 Day ML Challenge to learn and implement ML/DL concepts ranging from the basics to more advanced state of the art models.
Finished machine learning concepts from Andrew NG's course by Standford University on Coursera.

# Daily logs:

## Day-1 [17-09-2021] Introduction:
* Started ["Machine learning- A Probabilistic Perspective" by Kevin Murphy.](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf)
* Introduction to machine learning- learnt about: Matrix completion, Image Inpainting, collaborative filtering, No free lunch theorem and Market Basket analysis.
* 4 Distance measures widely used in machine learning.

## Day-2 [18-09-2021] Regression Analysis:
* Started by solving a problem related to the previous day which involved KNN and MNIST Dataset.
* Learnt about different types of regression namely Linear Regression, Logistic Regression, Ridge Regression, Lasso Regression and Polynomial Regression.
* Each one with their Equations and Graphs

## Day-3 [19-09-2021] Support Vector Machines:
![image](https://user-images.githubusercontent.com/77164319/133924595-894fdf2d-5a38-4ecc-a165-0712ba97e39a.png)
* Understood the intuition behind SVMs.
* Implemented a simple classification model using [scikit-learn](https://scikit-learn.org/stable/) SVM for the [Bank-retirement](https://www.kaggle.com/adarshkumarjha/bank-customer-retirement) dataset available on Kaggle.

##  Day-4 [20-09-2021] Naive-Bayes:
![image](https://user-images.githubusercontent.com/77164319/134170787-74e88010-a8b4-495e-aee8-4a4d0dfb368e.png)

* Understood the intuition behind Naive Bayes Classifier.
* Implemented a simple Naive Bayes Classification model using scratch for [Iris dataset](https://www.kaggle.com/vikrishnan/iris-dataset) available on Kaggle.

## Day-5 [21-09-2021] Hyperparameter tuning:
* learnt the importance of hyperparameter tuning in machine learning algorithms.
* saw different parameters for SVMs, Naive Bayes and KNNs. 

## Day-6 [22-09-2021] Bias, Variance, Cross validation and confusion matrix:
* Learnt concepts Bias, Variance, Cross Validation, Confusion Matrix.
* Implemented GridSearchCV and selected the best hyperparamter for Support Vector machine on a dataset.
* saw the [stanford cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-machine-learning-tips-and-tricks) for basics of machine learning.

## Day-7 [24-09-2021] Numpy, Pandas, Matplotlib and Seaborn:
* Went through the documentation and implemented topics.
* [Pandas cheatsheet](http://datacamp-community-prod.s3.amazonaws.com/f04456d7-8e61-482f-9cc9-da6f7f25fc9b)
* [Matplotlib Cheatsheet](http://datacamp-community-prod.s3.amazonaws.com/e1a8f39d-71ad-4d13-9a6b-618fe1b8c9e9)
* [Seaborn Cheatsheet](http://datacamp-community-prod.s3.amazonaws.com/263130e2-2c92-4348-a356-9ed9b5034247)

## Day-8 [11-10-2021] Kernels:
* Learnt basics of kernal functions.
* Also started watching Stanford's CS299 lecture on [kernels](https://www.youtube.com/watch?v=8NYoQiRANpg&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=10)

## Day-9 [12-10-2021] Kernels Continued:
* Learnt kernels for SVMs-polynomial and Radial Kernal(Radial bias function).
![image](https://user-images.githubusercontent.com/77164319/136872058-7ef4172d-f768-433f-b3f0-b0153eeed5e0.png)
* Learnt kernels and filters for convolution.
* Referred to chapter 14 in Machine learning-A Probabilistic perspective by Kevin Murphy.

## Day-10 [13-10-2021] Decision Trees:
* Learnt Basics of Decision Trees through numerous examples.
* learnt how to calculate gini index and other parameters.
* Watched Video by [StatQuest](https://www.youtube.com/watch?v=7VeUPuFGJHk) on Classification and regression Decision Trees.

## Day-11 [14-10-2021] Random Forest, Regression and Adaboost:
* Learnt the intuition behind random forest and adaboost.
* learnt concepts related to proximity matrix and distance matrix.
* How to combine different learning algorithms and average their results.
* Advantages and disadvantages of decision trees.
* Implemented and visualised a decision tree using [scikit learn](https://scikit-learn.org/stable/) and [seaborn](https://seaborn.pydata.org/).
* Learnt how to prune regresstion trees using the residual sum of squares and tree score.
* Learnt about the ID3 algorithm, Entropy and Information gain.
![image](https://user-images.githubusercontent.com/77164319/137328976-39dcfb02-476f-405e-9fd4-5c6b1c366fbd.png)

## Day 12 [15/10/2021] Neural Networks:
* Started Stanford's CS299 lecture on [Introduction to Neural Networks](https://www.youtube.com/watch?v=MfIjxPh6Pys&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=12&t=1854s).
* Learned about: Equational form of neurons and models.
* Terminology
* Advantages of neural networks.
* Softmax

## Day 13 [16/10/2021] Neural Networks and Backpropagation:
![image](https://user-images.githubusercontent.com/77164319/137585857-ebcd20dc-a30a-4084-a334-5f53e3733f08.png)
* Watched [StatQuest video](https://www.youtube.com/watch?v=IN2XmBhILt4) on very idea of Backpropagation.
* Learnt concepts of Chain rule in backpropagation and optimizing three parameters in a Neural Network simultaneously.

## Day 14 [18/10/2021] Neural Networks and Backpropagation Continued:
* Again analysed the concepts of Chain rule in backpropagation and optimizing three parameters in a Neural Network simultaneously.
* Learnt in depth the concepts of weights, bias, Gradient Descent and optimsation of parameters.

## Day 15 [19/10/2021] K-Nearest Neighbours:
* Introduction to K nearest neighbours algorithm.
* Applications and Advantages and disadvantages of KNNs.
* Learnt about hyperparameter tuning of K.

## Day 16 [20/10/2021] K-Nearest Neighbours Continued:
![image](https://user-images.githubusercontent.com/77164319/138191482-a82ae25e-f41d-4115-9a3e-c056b525c56f.png)

* Watched StatQuest video on KNN.
* Learnt concepts like Euclidean Distance and how exactly the KNN Algorithm works.

## Day 17 [26/10/2021] :
* Revised certain concepts

## Day 18 [27/10/2021] Activation functions, Optimization, Debugging Models:
* Learnt about various activation functions and which one to use for different models.
* Learnt about various optimizers.
* overfitting, underfitting and how to solve them.
* different cases with variance and bias and how to reach the appropriate training model.
* Learnt about data augmentation and how to implement it on deep learning models.

## Day 19 [30/10/2021] Project on implementation of KNN's
* Built a project on KNN that implements and uses Machine Learning in trading.
* [Reference](https://blog.quantinsti.com/machine-learning-k-nearest-neighbors-knn-algorithm-python/) for the project.

## Day 20 [31/10/21] Convolutional Neural Networks:
Some of the things I learned today:
* What are convolutional neural networks?
* What is the function of the CNN kernel?
* Continued to read up on ConvNet.
* Learned about the max pooling layer.

## Day 21 [1/11/21] Recurrent Neural Networks:
![image](https://user-images.githubusercontent.com/77164319/139613520-9dec5714-2310-419e-94da-4c66f34a022b.png)

Some of the things I learned today:
* What are [recurrent neural networks](https://www.youtube.com/watch?v=LHXXI4-IEns)?
* What makes RNNs more powerful than other architectures?
* Learned about the different RNNs architectures.
* Explored the different applications of RNNs.
* Learnt basics of [LSTM and GRU](https://www.youtube.com/watch?v=8HyCNIVRbSU). learnt the intuition behind them

## Day 22 [2/11/21] Long Short Term memory:
Learnt the following today:

* basics of [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
* The Core Idea Behind LSTMs(Long Short Trem Memory)
* Step-by-Step LSTM Walk Through
* Variants on Long Short Term Memory

## Day 23 [3/11/21] Gated recurrent units:
* Learnt about update gate and reset gate.
* basic understanding of the hidden state.
* how to execute [GRUs](https://www.youtube.com/watch?v=Ogvd787uJO8) using keras.
* read about [working of gates and GRUs.](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-gated-recurrent-unit-gru/)

## Day 24 [4/11/21] Project on RNNs:
* Worked on sequence prediction problem using RNN.
* [Reference](https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/) for the project

## Day-25 [5/11/21] Vanishing gradient problem:
* In machine learning, the [vanishing gradient problem](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484) is encountered when training artificial neural networks with gradient-based learning methods and backpropagation.
* Learnt the disadvantages of it.
* Learnt how to fix the [vanishing gradient problem using ReLU.](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)

## Day-26 [6/11/2021] Basics of NLP (Natural Language Processing):
Learnt basics of :

* NLP and its applications in various domains.
* Stop Words
* Sentence and Word Tokenization.
* Text Stemming
* Text Lemmatization
* Regex
* Bag-of-Words

## Day-27 [7/11/2021] Vectorizers :
Learn the basics of various types of [vectorizers](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide):

* Count Vectorizer
* Tf-Idf
* Word2Vec

## Day-28 [8/11/2021] Project on NLP :
![image](https://user-images.githubusercontent.com/77164319/140771312-f4d95aa3-cdcf-4c6e-962d-8bda9bd8a898.png)

* Worked on Sentiment Analysis using NLP.
* Sentiment analysis refers to the application of natural language processing, computational linguistics, and text analysis to identify and classify subjective opinions in source documents.
* [Reference for the project](https://thecleverprogrammer.com/2020/12/07/sentiment-analysis-with-python/).

## Day-29 [9/11/2021] Exploratory Data Analysis:

### What is EDA?

![image](https://user-images.githubusercontent.com/77164319/140796276-5a29139f-ca7f-46d1-b425-88c86c2bd831.png)

In statistics, [exploratory data analysis](https://www.analyticsvidhya.com/blog/2021/04/rapid-fire-eda-process-using-python-for-ml-implementation/) is an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods
* Learnt about different proprocessing techniques on a dataset using pandas and numpy.
* Learn what [plots are significant](https://chartio.com/learn/charts/essential-chart-types-for-data-visualization/) to different types of data.
* Learnt how to identify outliers.
* Learnt about various techniques to create models for predicting certain kind of outputs.
* How to make inferences and write a detailed report.

## Day-30 [10/11/2021] Batch, Mini Batch and Stochastic Gradient Desent :

![image](https://user-images.githubusercontent.com/77164319/141031741-f66bdc77-6ebe-48c7-aead-d4fd36ee954b.png)

* Learnt about [Batch](https://www.youtube.com/watch?v=sDv4f4s2SB8) Gradient Desent.
* Learnt [Stochastic](https://www.youtube.com/watch?v=vMh0zPT0tLI) Gradient Desent.
* Learnt major [differences between Batch Gradient Desent and Stochastic Gradient Desent](https://www.geeksforgeeks.org/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/).
* Other [References](https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a).


## Day-31 [11/11/2021] Introduction to K-means clustering :
* intuition behind [K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
* Interesting visualization of [K-means clustering](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/)
* [Hyperparameter tuning](https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/) in K-means clustering algorithm

## Day-32 [12/11/2021] K-means clustering continued:

![image](https://user-images.githubusercontent.com/77164319/141470472-b2e18552-dc15-4bb2-b054-86c9e1b7007c.png)

* Learnt some of the applications of K-means clustering.
* Learnt the [Math](https://muthu.co/mathematics-behind-k-mean-clustering-algorithm/) behind K-means clustering.

## Day-33 [13/11/2021] K-means clustering mini project:

* Using Python (Pandas, NumPy) to gather and assess the data and scikit-learn to train a K-Means model to detect if a banknote is genuine or forged.
* [Reference](https://towardsdatascience.com/k-means-clustering-project-banknote-authentication-289cfe773873) for the project

## Day-34 [14/11/2021] Momentum Optimiser:
* Watched [DeepLearningAl video](https://www.youtube.com/watch?v=k8fTYJPd3_I) on Gradient Descent with momemtum.
* Learnt the function of each parameter (especially the role of β in both (β) and (1 - β)) that is extremely useful in gradient descent.
* Read an [article](https://medium.com/@vinodhb95/momentum-optimizer-6023aa445e18) on momemtum optimiser.

## Day-35 [15/11/21] Hopfield Networks: 
Learned about:
* What [Hopfield networks](https://medium.com/@serbanliviu/hopfield-nets-and-the-brain-e5880070cdba) are.
* Learnt about the Hebbian postulate: "Neurons that fire together wire together, neurons out of sync, fail to link."
* Learnt about the energy function of the Hopfield Nets.
* How it is similar to our brain.
* Advantages of Hopfield Nets.
* How to use Hopfield networks.
* How Hopfield networks improve on the RNN model.


## Day-36 [16/11/2021] Boltzmann Machines:
* Learnt what Boltzmann machines are.
* Learnt types of Boltzmann Machines and working of them.
* Learnt:
  1.  Restricted Boltzmann Machines (RBMs)
  2. Deep Belief Networks (DBNs)
  3.  Deep Boltzmann Machines (DBMs)

## Day- 37 [17/11/2021] Autoencoders:
Learned about:
* What [Autoencoder networks](https://www.jeremyjordan.me/autoencoders/) are.
* How an Autoecoder functions.
* The components that make up an Autoencoder.
* Applications of Autoencoders.



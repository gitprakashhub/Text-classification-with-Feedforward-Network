# Text-classification-with-Feedforward-Network

The goal of this assignment is to develop a Feedforward neural network for text classification.

For that purpose, you will implement:

    Text processing methods for transforming raw text data into input vectors for your network

    A Feedforward network consisting of:
        One-hot input layer mapping words into an Embedding weight matrix
        One hidden layer computing the mean embedding vector of all words in input followed by a ReLU activation function
        Output layer with a softmax activation.

    The Stochastic Gradient Descent (SGD) algorithm with back-propagation to learn the weights of your Neural network. Your algorithm should:
        Use (and minimise) the Categorical Cross-entropy loss function
        Perform a Forward pass to compute intermediate outputs
        Perform a Backward pass to compute gradients and update all sets of weights
        Implement and use Dropout after each hidden layer for regularisation

    Discuss how did you choose hyperparameters? You can tune the learning rate (hint: choose small values), embedding size {e.g. 50, 300, 500}, the dropout rate {e.g. 0.2, 0.5} and the learning rate. Please use tables or graphs to show training and validation performance for each hyperparameter combination.

    After training a model, plot the learning process (i.e. training and validation loss in each epoch) using a line plot and report accuracy. Does your model overfit, underfit or is about right?.

    Re-train your network by using pre-trained embeddings (GloVe) trained on large corpora. Instead of randomly initialising the embedding weights matrix, you should initialise it with the pre-trained weights. During training, you should not update them (i.e. weight freezing) and backprop should stop before computing gradients for updating embedding weights. Report results by performing hyperparameter tuning and plotting the learning process. Do you get better performance?

    Extend you Feedforward network by adding more hidden layers (e.g. one more or two). How does it affect the performance? Note: You need to repeat hyperparameter tuning, but the number of combinations grows exponentially. Therefore, you need to choose a subset of all possible combinations

    Provide well documented and commented code describing all of your choices. In general, you are free to make decisions about text processing (e.g. punctuation, numbers, vocabulary size) and hyperparameter values. We expect to see justifications and discussion for all of your choices.

    Provide efficient solutions by using Numpy arrays when possible. Executing the whole notebook with your code should not take more than 10 minutes on any standard computer (e.g. Intel Core i5 CPU, 8 or 16GB RAM) excluding hyperparameter tuning runs and loading the pretrained vectors. You can find tips in Intro to Python for NLP.

Data

The data you will use for the task is a subset of the AG News Corpus and you can find it in the ./data_topic folder in CSV format:

    data_topic/train.csv: contains 2,400 news articles, 800 for each class to be used for training.
    data_topic/dev.csv: contains 150 news articles, 50 for each class to be used for hyperparameter selection and monitoring the training process.
    data_topic/test.csv: contains 900 news articles, 300 for each class to be used for testing.

Pre-trained Embeddings

You can download pre-trained GloVe embeddings trained on Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) from here. No need to unzip, the file is large.

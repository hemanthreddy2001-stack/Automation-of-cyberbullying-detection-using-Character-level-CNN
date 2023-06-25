# Automation-of-cyberbullying-detection-using-Character-level-CNN
# Python script that uses the tkinter library to create a graphical user interface (GUI) for a cyberbullying detection system. Here is a breakdown of the code:

# main.py:
The code begins by importing various modules from the `tkinter`, `matplotlib`, `nltk`, `keras`, `string`, `os`, and `pandas` libraries.
It creates a tkinter window object and sets its title and geometry.
Several global variables are declared to store information related to the model, dataset, and evaluation metrics.


The code defines several functions that will be used in the GUI:
`uploadDataset()`: Allows the user to select a dataset file using a file dialog.
`clean_doc(doc)`: Cleans a given text document by removing punctuation, stopwords, and non-alphabetic characters.
`getCharacters(tokens)`: Converts a given text into a sequence of characters.
`cleancodeVec()`: Processes the dataset by cleaning the text messages, generating word and character-level representations, and storing them in respective arrays.
`max_length(lines)`: Calculates the maximum length of a list of lines (in terms of words).
`encode_text(tokenizer, lines, length)`: Encodes a list of lines using a tokenizer and pads the sequences to a specified length.
`create_tokenizer(lines)`: Creates a tokenizer object and fits it on a given list of lines.
`calculateWordMetrics()`: Evaluates the word-based model's performance metrics on a test set.
`wordCNN()`: Trains a word-based CNN model using the processed dataset and displays the model summary and performance metrics.
`calculateCharMetrics()`: Evaluates the character-based model's performance metrics on a test set.
`create_char_tokenizer(lines)`: Creates a character-level tokenizer object and fits it on a given list of lines.
`charCNN()`: Trains a character-based CNN model using the processed dataset and displays the model summary and performance metrics.

The GUI layout is defined using tkinter widgets, such as labels, buttons, and a text area, and their respective functionalities are assigned to the defined functions.
The main tkinter event loop is started to display the GUI and handle user interactions.


# `history.pckl` file:
The `history.pckl` file is used to store the training history of the word-based CNN model. Here is a detailed explanation of its functionality:

The `history.pckl` file is created using the pickle module, which allows for easy serialization and deserialization of Python objects.

After training the word-based CNN model (word_model), the training history is obtained from the `hist.history` attribute. This history contains various metrics such as accuracy, loss, validation accuracy, and validation loss at each epoch during training.

The training history is then stored in the history.pckl file using the `pickle.dump()` function. This function serializes the Python object (in this case, the training history) and writes it to the file.

Storing the training history allows you to analyze and visualize the model's performance over time. For example, you can plot the training and validation accuracy/loss curves to assess the model's convergence and identify any overfitting or underfitting issues.

When the model is reloaded in the future, the history.pckl file can be loaded using the `pickle.load()` function. This function reads the serialized object from the file and reconstructs it as a Python object.

By loading the training history, the recorded metrics for further analysis or reporting can be accessed and utilized. For example, to display the accuracy, precision, recall, and F1-score values achieved by the model during training.


# `model_weights.h5` file:
The `model_weights.h5` file is used to save and load the trained weights of the word-based CNN model. Here is an explanation of its use:

After training the word-based CNN model (word_model), the trained weights of the model are saved to the model_weights.h5 file using the `save_weights()` method. This method is provided by `Keras`, a popular deep learning library, and it allows for saving the weights of a model to a file.

Saving the model weights is important because it allows you to preserve the learned parameters of the model. These parameters represent the knowledge acquired by the model during the training process and are essential for making predictions on new data.

The model_weights.h5 file is saved in the `Hierarchical Data Format (HDF5)` format. HDF5 is a data model, library, and file format for storing and managing large and complex datasets.

Saving the model weights separately from the model architecture (which is saved in the `model.json` file) allows for flexibility. It allows reuse of architecture and load different sets of weights for various purposes, such as fine-tuning the model on new data or deploying the model for inference.

To load the trained weights from the model_weights.h5 file in the future, you can use the `load_weights()` method. This method loads the weights into the model from the specified file, allowing you to restore the model's state as it was after training.

Loading the trained weights is crucial for reusing the trained model without the need for retraining. By loading the saved weights, you can directly use the model to make predictions on new data, leveraging the knowledge acquired during training.

# Datasets:
The provided code uses two datasets: `train_data` and `test_data`. 

train_data: This dataset represents the training data used to train the word-based CNN model. It is a collection of text samples, where each sample consists of a tweet and its corresponding sentiment label. The sentiment label indicates whether the sentiment of the tweet is positive, negative, or neutral. The train_data dataset is a list of tuples, where each tuple contains the tweet text as a string and its sentiment label as an integer.

Example format of train_data:

`[  ("I love this movie!", 1),  ("The weather today is terrible.", -1),  ("I'm feeling neutral about this.", 0),  ...]`

In the example above, the sentiment label 1 indicates a positive sentiment, -1 indicates a negative sentiment, and 0 indicates a neutral sentiment.

During the training process, the model learns from these labeled samples to predict the sentiment of unseen tweets.


test_data: This dataset represents the testing data used to evaluate the performance of the trained model. Similar to train_data, test_data is a collection of text samples with their corresponding sentiment labels. It follows the same format as train_data, where each sample is a tuple containing the tweet text and its sentiment label.

Example format of test_data:

`[  ("This movie is amazing!", 1),  ("I'm not happy with the service.", -1),  ("It's neither good nor bad.", 0),  ...]`

The test_data dataset is used to assess how well the trained model generalizes to unseen data by predicting the sentiment labels for the test samples.

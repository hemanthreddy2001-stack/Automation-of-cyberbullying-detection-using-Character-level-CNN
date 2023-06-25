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

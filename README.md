# Spam-Detection-Using-Model-Using-PyTorch
This project implements a Spam Detection Model using the PyTorch deep learning framework. The goal of the model is to classify text messages as either spam or ham (not spam) based on their content. The dataset consists of labeled SMS messages, and the classification is achieved using a Convolutional Neural Network (CNN) architecture. Below is a detailed breakdown of the various steps and components used in this project:

# 1. Library Imports
The project begins by importing the necessary libraries and frameworks:
Pandas is used for loading and processing the dataset.
PyTorch (including its submodules like torch.nn, torch.optim, and torch.utils.data) is the deep learning framework used for building, training, and evaluating the model.
NLTK (Natural Language Toolkit) is used for text preprocessing (tokenization, stopword removal).
scikit-learn is used for splitting the data into training and test sets, as well as for encoding the labels.
Collections (specifically Counter) is used to build the vocabulary from the dataset.

# 2. NLTK Data Downloads
Since the model involves text preprocessing, necessary data is downloaded using nltk.download:
'punkt': Used for tokenizing sentences into words.
'stopwords': Used to filter out common words that do not contribute much to spam detection, such as "the", "is", etc.

# 3. Dataset Loading and Preparation
The dataset used in this project is stored in a CSV file, which contains two important columns:
Label (v1): Indicates whether a message is spam or ham.
Message (v2): Contains the text of the SMS message.
The dataset is loaded using Pandas, and the two columns (v1, v2) are renamed as 'label' and 'message'. This step simplifies further processing. After that:
Label Encoding: The labels (spam and ham) are converted into numeric form using LabelEncoder. Here, spam = 1 and ham = 0.
A quick check is performed to see the balance of spam and ham messages.

# 4. Text Preprocessing
To process the text data into a form that the neural network can understand, the following steps are applied:
Tokenization: Each message is tokenized into individual words using word_tokenize from NLTK.
Lowercasing and Filtering: All words are converted to lowercase, and only alphabetic words are retained (i.e., numbers and punctuation are removed).
Stopword Removal: Common words (stopwords) that do not contribute much to determining whether a message is spam or ham are removed using a predefined set of stopwords from NLTK.
Each message is transformed into a list of tokens (i.e., words) after this step.

# 5. Building Vocabulary and Encoding Messages
A vocabulary of unique words is built from the preprocessed messages. Each word is assigned a unique integer ID to encode messages into a sequence of numbers. This is necessary since the neural network can only work with numeric inputs. The process involves:
Counting words in the entire dataset using Counter to create the vocabulary.
Mapping each word to an integer ID starting from 1. This creates a mapping dictionary where the word is the key and the integer is the value.
Encoding each message: Each message is then encoded as a sequence of integer IDs based on this vocabulary.

# 6. Padding Sequences
Since different messages have different lengths, all sequences are padded to the length of the longest message. This ensures that all input sequences have the same length, which is required by the CNN model:
Messages shorter than the maximum length are padded with zeros.

# 7. Splitting Dataset into Training and Test Sets
The dataset is split into a training set (80%) and a test set (20%) using train_test_split from scikit-learn:
X: The input sequences (padded and encoded messages).
y: The labels (spam or ham). These are converted to PyTorch tensors, which are the standard data structure used in PyTorch models.

# 8. Creating Dataset Class
A custom SpamDataset class is defined to work with the PyTorch DataLoader class. This class is responsible for:
Returning input-output pairs (message, label) when indexed.
Providing the length of the dataset.
The dataset is then loaded into training and test loaders, which allow efficient loading of data in batches during model training.

# 9. CNN Model Definition
The core of this project is a Convolutional Neural Network (CNN), which is used for text classification. The architecture of the model includes:
Embedding Layer: Converts integer-encoded words into dense vectors of fixed size (embed_size = 128).
Convolutional Layers: Multiple convolutional layers with different filter sizes (3, 4, 5) extract features from sequences of word embeddings. This allows the model to capture word patterns of different lengths.
Max Pooling: Each convolutional layer is followed by a max pooling layer that reduces the dimensionality by selecting the most important feature for each filter.
Fully Connected Layer: After feature extraction, the outputs from the convolutional layers are concatenated and passed through a fully connected layer to classify the message as spam or ham.
Dropout Layer: A dropout layer is used to reduce overfitting by randomly dropping neurons during training.

# 10. Model Training
The model is trained using the following process:
Loss Function: Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss), which is commonly used for binary classification tasks.
Optimizer: Adam optimizer (optim.Adam), which adjusts the model weights during training based on the calculated loss.
Training Loop: The model is trained for 5 epochs. In each epoch:
A forward pass computes the model predictions.
The loss is calculated using the predictions and actual labels.
Backpropagation updates the model weights to minimize the loss.
The training loop prints the average loss after each epoch to monitor the model's progress.

# 11. Model Evaluation
The model is evaluated on the test set to measure its accuracy. This involves:
Model Evaluation Loop: The model is put into evaluation mode, and the test data is passed through the network. Predictions are made using the sigmoid function to map the output logits to probabilities.
Accuracy Calculation: The number of correct predictions (spam or ham) is compared to the actual labels, and the accuracy is reported as a percentage.

# 12. Prediction Function
A predict_message function is defined to predict whether a new message is spam or ham. It follows these steps:
Preprocess the message: The message is tokenized, stopwords are removed, and it is encoded as a sequence of integers.
Padding: The sequence is padded to match the maximum length used during training.
Prediction: The preprocessed message is passed through the model, and the output is classified as spam if the sigmoid probability is greater than 0.5; otherwise, it is classified as ham.
An example message is provided for testing the function.

# Conclusion:
This project demonstrates how to build a spam detection model using a Convolutional Neural Network in PyTorch. By preprocessing text data, building a vocabulary, encoding messages, and leveraging CNN layers for feature extraction, the model can effectively classify messages as spam or ham. The modular design of the code allows it to be easily extended or integrated with other applications.

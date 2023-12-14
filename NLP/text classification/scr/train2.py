import io
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import config
import dataset
import engine
import lstm

def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
    data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    """
    # initialize matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    # loop over all the words
    for word, i in word_index.items():
        # if word is found in pre-trained embeddings, update the matrix.
        # if the word is not found, the vector is zeros!
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]

        # return embedding matrix
        return embedding_matrix

def run(df, fold):
    """
    Run training and validation for a given fold and dataset
    :param df: pandas dataframe with kfold column
    :param fold: current fold, int
    """
    # fetch training and validation dataframe
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    print("Fitting tokenizer")
    # initialize CountVectorizer with NLTK's word_tokenize
    count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

    # fit count_vec on training data reviews
    count_vec.fit(train_df.review)

    # transform training and validation data reviews
    xtrain = count_vec.transform(train_df.review)
    xtest = count_vec.transform(valid_df.review)

    # initialize dataset class for training and validation
    train_dataset = dataset.IMDBDataset(reviews=xtrain,
                                        targets=train_df.sentiment.values)
    valid_dataset = dataset.IMDBDataset(reviews=xtest,
                                        targets=valid_df.sentiment.values)

    # create torch dataloader for training and validation
    # torch dataloader loads the data using dataset
    # class in batches specified by batch size
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config.TRAIN_BATCH_SIZE,
                                                    num_workers=2)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=config.VALID_BATCH_SIZE,
                                                    num_workers=1)

    print("Loading embeddings")
    # load embeddings as shown previously
    embedding_dict = load_vectors("../input/crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(count_vec.vocabulary_, embedding_dict)

    # create torch device, since we use gpu, we are using cuda
    device = torch.device("cuda")

    # fetch our LSTM model
    model = lstm.LSTM(embedding_matrix)

    # send model to device
    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Model")
    # set best accuracy to zero
    best_accuracy = 0
    # set early stopping counter to zero
    early_stopping_counter = 0

    # train and validate for all epochs
    for epoch in range(config.EPOCHS):
        # train one epoch
        engine.train(train_data_loader, model, optimizer, device)
        # validate
        outputs, targets = engine.evaluate(valid_data_loader, model, device)

        # use threshold of 0.5
        # please note we are using linear layer and no sigmoid
        # you should do this 0.5 threshold after sigmoid
        outputs = np.array(outputs) >= 0.5

        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, outputs)
        print(
            f"FOLD:{fold}, Epoch: {epoch}, Accuracy Score = {accuracy}"
        )

        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1

        if early_stopping_counter > 2:
            break

if __name__ == "__main__":
    # load data
    df = pd.read_csv("../input/imdb_folds.csv")
    # train for all folds
    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)
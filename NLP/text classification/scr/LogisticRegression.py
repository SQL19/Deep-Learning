import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

def stratifiedKFold(df):
    # create a new column kfold and dill it with -1
    df['kfold'] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.sentiment.values

    # create k folds
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    return df

if __name__ == '__main__':
    # read the dataset
    df = pd.read_csv('../input/imdb.csv')

    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)

    # create stratified k folds
    df = stratifiedKFold(df)

    # go over k folds
    for fold_ in range(5):
        df_train = df[df.kfold != fold_].reset_index(drop=True)
        df_test = df[df.kfold == fold_].reset_index(drop=True)

        # initialize CountVectorizer with NLTK's word_tokenize
        count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

        # fit count_vec on training data reviews
        count_vec.fit(df_train.review)

        # transform training and validation data reviews
        xtrain = count_vec.transform(df_train.review)
        xtest = count_vec.transform(df_test.review)

        # initialize logistic regression model
        model = linear_model.LogisticRegression()

        # fit the model on training data reviews and sentiment
        model.fit(xtrain, df_train.sentiment)

        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)

        # calculate accuracy
        accuracy = metrics.accuracy_score(df_test.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")

import logging

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import pandas as pd
from sklego.preprocessing import PandasTypeSelector
import pickle
from contextlib import redirect_stdout
import sys
from numpy import nan

logger = logging.getLogger(__name__)

"""
Refs:
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
"""


def predict_unseen(data, pipeline):
    indices = [passenger['Name'] for passenger in data]
    data = preprocess_data(pd.DataFrame(data).replace('unknown', nan))
    preds = pipeline.predict(data)
    result = [str(p) for p in preds]
    return {name: result[i] for i, name in enumerate(indices)}

def load_model(model_path):
    with open(model_path, "rb") as pickle_file:
        return pickle.load(pickle_file)


def load_data(data_path="/opt/ml/input/data/training/titanic.csv"):
    # df = pd.read_csv(sys.stdin, sep="\t")
    df = pd.read_csv(data_path, sep="\t")
    return df.drop(columns="Survived"), df["Survived"]


def preprocess_data(X):
    return X.drop(columns=["Name", "PassengerId", "Survived"], errors='ignore').assign(
        Sex=lambda d: d["Sex"].fillna("unknown"),
        Parch=lambda d: d["Parch"].fillna("unknown"),
        Ticket=lambda d: d["Ticket"].fillna("unknown"),
        Cabin=lambda d: d["Cabin"].fillna("unknown"),
        Embarked=lambda d: d["Embarked"].fillna("unknown"),
    )


def train_model(X, y):
    logger.info(f"fitting model on optml with shapes {X.shape}, {y.shape}")

    numerical_pipeline = make_pipeline(
        PandasTypeSelector("number"),
        SimpleImputer(), StandardScaler()
    )
    object_pipeline = make_pipeline(
        PandasTypeSelector("object"),
        # handle_unknown = 'ignore' voorkomt dat categories zonder label
        # in CV splits een probleem gaan vormen.
        OneHotEncoder(sparse=False, handle_unknown="ignore")
    )
    pipeline = make_pipeline(
        make_union(numerical_pipeline,
                   object_pipeline),
        LogisticRegression()
    )

    # Redirecting stdout because sklearn uses stdout for logging,
    # and we output our model to the std out, so we can pipe it.
    with redirect_stdout(sys.stderr):
        # gs = GridSearch()
        pipeline.fit(X, y)

        return pipeline


if __name__ == "__main__":
    X, y = load_data("optml/input/data/training/titanic.csv")
    X = preprocess_data(X)
    fitted_model = train_model(X, y)
    print(pickle.dumps(fitted_model))

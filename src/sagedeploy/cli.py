import click
import logging
import pickle
from sagedeploy.model import load_data, preprocess_data, train_model

import subprocess

DEFAULT_MODEL_PATH = '/opt/ml/model/model.pkl'
DEFAULT_INPUT_PATH = '/opt/ml/input/data/training/titanic.csv'

logger = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', is_flag=True)
def main(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)


@main.command()
@click.option('-p', '--path', default=DEFAULT_INPUT_PATH)
@click.option('-o', '--output-path', default=DEFAULT_MODEL_PATH)
def train(path, output_path):
    logger.info("Starting training procedure")
    X, y = load_data(data_path=path)
    X = preprocess_data(X)
    model = train_model(X, y)

    # Sagemaker will take care of time stamping, so we only need
    # to dump the model file in the default model directory
    with open(output_path, 'wb') as file:
        pickle.dump(model, file)


@main.command()
@click.option('-m', '--model-path', default=DEFAULT_MODEL_PATH)
def serve(model_path):
    process = subprocess.Popen(["gunicorn", f"wsgi:create_app(\"{model_path}\")", "-b", "0.0.0.0:8080"])
    process.wait()
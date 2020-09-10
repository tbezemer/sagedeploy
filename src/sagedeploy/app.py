from flask import Flask, request
from sagedeploy.model import predict_unseen, load_model
import logging

logger = logging.getLogger(__name__+":endpoint")
logging.basicConfig(level=logging.INFO)

def create_app(model_path):
    app = Flask(__name__ + ":endpoint")
    pipeline = load_model(model_path)
    @app.route('/invocations', methods=['POST'])
    def infer():
        """
        Process a request:
        POST application/JSON {
        ...
        'data' : {feature1:.. feature2:.. feature3: ..}
        }
        :return:
        """
        request_body = request.json
        data = request_body['data'] if isinstance(request_body['data'], list) else [request_body['data']]

        logger.info(f"Predicting {len(data)} items.")
        return predict_unseen(data, pipeline), 200

    @app.route("/ping", methods=['GET'])
    def ping():
        return {}, 200

    return app

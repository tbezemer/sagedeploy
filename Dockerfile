FROM python:3.8-slim

WORKDIR sagedeploy

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

ENTRYPOINT ["sagedeploy"]
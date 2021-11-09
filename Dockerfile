FROM python:3.7-slim
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt && rm -rf requirements.txt
COPY app.py app.py
# CMD mlflow experiments create --experiment-name iris \
#    && mlflow experiments create --experiment-name wine \
#    && mlflow experiments create --experiment-name diabetes

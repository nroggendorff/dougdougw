FROM python:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /.cache && chmod 777 /.cache

RUN mkdir /app/model && chmod 777 /app/model

CMD ["python3", "train_and_test.py"]
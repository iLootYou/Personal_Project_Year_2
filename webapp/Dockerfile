FROM python:3.11-alpine

WORKDIR /webapp

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
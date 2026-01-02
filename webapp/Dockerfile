FROM python:3.11-slim

WORKDIR /webapp

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]

# Run the following command in the same directory as this file to build image: build -t match-predictor .
# To run the container: docker run -p 5000:5000 match-predictor
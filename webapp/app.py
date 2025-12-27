from flask import Flask, render_template, jsonify, request, redirect, url_for, session

app = Flask(__name__)

@app.route("/")
def predict():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

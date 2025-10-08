from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/api/predict', method=('GET'))
def predict_diabetes():
    return 0

if __name__ == '__main__':
    app.run(
        port=1249,
        debug=True
    )


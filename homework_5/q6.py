import pickle

from flask import Flask
from flask import request
from flask import jsonify


def load(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


dv = load('dv.bin')
model = load('model2.bin')

app = Flask('hw_5')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    x = dv.transform([client])
    y_pred = model.predict_proba(x)[0, 1]
    get_result = y_pred >= 0.5

    result = {
        'get_result_probability': float(y_pred),
        'get_result': bool(get_result)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
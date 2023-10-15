import pickle


def load(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


dv = load('dv.bin')
model1 = load('model1.bin')

client = {"job": "retired", "duration": 445, "poutcome": "success"}

x = dv.transform([client])
y_pred = model1.predict_proba(x)[0, 1]

print(y_pred)  # 0.902
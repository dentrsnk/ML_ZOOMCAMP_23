import requests


url = "http://localhost:9696/predict"

client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
response = requests.post(url, json=client).json()

print(response) # 0.13968947052356817
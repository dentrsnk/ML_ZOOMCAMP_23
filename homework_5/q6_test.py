import requests


url = "http://localhost:9696/predict"

client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url, json=client).json()

print(response)  # 0.9019309332297606
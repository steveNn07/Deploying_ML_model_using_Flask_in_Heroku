import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'RM':7, 'LSTAT':19, 'PTRATIO':15})

print(r.json())
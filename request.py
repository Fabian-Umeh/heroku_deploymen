import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Year':2, 'Methane':9, 'ccgo':6, 'Gdp_per_capita':6})

print(r.json())
import requests

url = "http://127.0.0.1:5002/predict"
data = {
    "lat": 27.83,
    "lon": 85.58,
    "target_date": "2021-06-15"
}

try:
    resp = requests.post(url, json=data).json()
    print("API Response for Melamchi (2021-06-15):")
    print(resp)
except Exception as e:
    print("API not running or error:", e)

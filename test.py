import requests
import json

url = "http://localhost:8000/api/search"
payload = {
    "query": "wireless headphones",
    "limit": 5
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise HTTP errors
    results = response.json()
    print(json.dumps(results, indent=2))
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
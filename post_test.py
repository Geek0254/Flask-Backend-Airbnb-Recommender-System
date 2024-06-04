import requests

url = "http://127.0.0.1:5000/recommend"
data = {
    "city": "nyc",
    "user_preferences": {
        "neighbourhood_cleansed": "Kensington",
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "min_price": 500,
        "max_price": 1000,
        "min_reviews": 10,
        "min_availability": 50,
        "min_rating": 4.0,
        "min_nights": 2,
        "min_beds": 1,
        "min_bedrooms": 1
    }
}

response = requests.post(url, json=data)
try:
    response_data = response.json()
    print(response_data)
except requests.exceptions.JSONDecodeError:
    print("Response content is not valid JSON:")
    print(response.content)

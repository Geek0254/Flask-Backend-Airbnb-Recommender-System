import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# City mapping for directories
city_directories = {
    'nyc': 'NYC',
    'berlin': 'Berlin',
    'amsterdam': 'Amsterdam',
    'sydney': 'Sydney',
    'rome': 'Rome',
    'tokyo': 'Tokyo',
    'barcelona': 'Barcelona',
    'brussels': 'Brussels'
}

def load_data(city):
    # Load the dataset based on the city
    directory = city_directories.get(city.lower())
    if not directory:
        raise ValueError("City not found in directory mapping.")
    file_path = f'./datasets/{directory}/{city.lower()}_airbnb_listings.csv'
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

def preprocess_data(data):
    # Handle missing values
    data.fillna({
        'reviews_per_month': data['reviews_per_month'].median(),
        'last_review': data['last_review'].mode()[0],
        'host_name': 'N/A',
        'neighbourhood_cleansed': 'N/A',
        'neighbourhood_group_cleansed': 'N/A',
        'house_rules': 'N/A',
        'host_response_time': 'N/A',
        'host_response_rate': 'N/A',
        'property_type': 'N/A',
        'beds': data['beds'].median(),
        'bedrooms': data['bedrooms'].median(),
        'instant_bookable': 'N/A'
    }, inplace=True)

    # Convert and clean columns
    data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)
    data['last_review'] = pd.to_datetime(data['last_review'], errors='coerce')
    data['instant_bookable'] = data['instant_bookable'].replace({'t': 'yes', 'f': 'no', 'N/A': 'N/A'})

    return data

def train_and_evaluate_models(data):
    # Prepare features and target variable for KNN
    knn_features = ['price', 'number_of_reviews', 'availability_365', 'minimum_nights', 'reviews_per_month', 'beds', 'bedrooms']
    X = data[knn_features].fillna(0)
    y = data['review_scores_rating'].fillna(0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN model
    knn = KNeighborsRegressor(n_neighbors=3, p=1)
    knn.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = knn.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"KNN Model Mean Squared Error: {mse}")
    print(f"KNN Model R² Score: {r2}")

    # Save KNN model and scaler
    with open(f'./models/knn_model_evaluation_{city}.pkl', 'wb') as file:
        pickle.dump(knn, file)
    with open(f'./models/scaler_model_evaluation_{city}.pkl', 'wb') as file:
        pickle.dump(scaler, file)

def train_tfidf_evaluate(data):
    # Train TF-IDF Vectorizer
    cbf_features = ['neighbourhood_cleansed', 'room_type', 'property_type']
    data['combined_features'] = data[cbf_features].astype(str).apply(lambda x: ' '.join(x), axis=1)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data['combined_features'])

    # Save TF-IDF Vectorizer
    with open(f'./models/cbf_model_evaluation_{city}.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

if __name__ == "__main__":
    # List of available cities
    available_cities = list(city_directories.keys())

    # User input for city
    print("Available cities:")
    for idx, city in enumerate(available_cities):
        print(f"{idx + 1}. {city.capitalize()}")

    try:
        choice = int(input("Enter the number corresponding to the city you want to evaluate: "))
        if choice < 1 or choice > len(available_cities):
            raise ValueError("Invalid choice.")
        city = available_cities[choice - 1]
        print(f"Evaluating models for {city.capitalize()}...")

        # Load, preprocess, train, and evaluate models
        data = load_data(city)
        data = preprocess_data(data)
        train_and_evaluate_models(data)
        train_tfidf_evaluate(data)

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")

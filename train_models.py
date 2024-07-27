import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import requests

# Google Drive direct download links
file_links = {
    'nyc': 'https://drive.google.com/uc?export=download&id=1U5LY2d6MPwVVjhECl5m8MpFpo-bDPLxj',
    'berlin': 'https://drive.google.com/uc?export=download&id=13MaITpBsID3lHpPadgDIMFJNNDs4nA5l',
    'amsterdam': 'https://drive.google.com/uc?export=download&id=1MAB_koP-CPDkhjVs3DUbeU_jLuD3HCMD',
    'sydney': 'https://drive.google.com/uc?export=download&id=1l0MY9e3FoV-qNkD9arAdjo8qIXvy4H0T',
    'rome': 'https://drive.google.com/uc?export=download&id=1ua6XGqqGCNq6yHBtFYxo6_DuH9sOfURD',
    'tokyo': 'https://drive.google.com/uc?export=download&id=1CI_JxuNs2ZY0rCHiw_YoE8SJg8lLME1w',
    'barcelona': 'https://drive.google.com/uc?export=download&id=1vGDtFTscAXPHdtFHcbOBo47NW_qzEDSd',
    'brussels': 'https://drive.google.com/uc?export=download&id=19EaSibkDr7VGdUd5Afo_JF1rMe_8MoEy'
}

def download_file_from_google_drive(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

def load_data(city):
    file_path = f'./datasets/{city}_airbnb_listings.csv'
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_link = file_links.get(city)
        if not file_link:
            print(f"No file link found for city: {city}")
            return None
        download_file_from_google_drive(file_link, file_path)
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data for {city}: {e}")
        return None

def train_and_save_models(city):
    # Load dataset
    data = load_data(city)
    if data is None:
        print(f"Data for city {city} could not be loaded.")
        return

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

    # Convert date columns to datetime
    data['last_review'] = pd.to_datetime(data['last_review'], errors='coerce')

    # Clean price column
    data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)

    # Replace instant bookable values
    data['instant_bookable'] = data['instant_bookable'].replace({'t': 'yes', 'f': 'no', 'N/A': 'N/A'})

    # Train KNN model
    knn_features = ['price', 'number_of_reviews', 'availability_365', 'minimum_nights', 'reviews_per_month', 'beds', 'bedrooms']
    X = data[knn_features].fillna(0)
    y = data['review_scores_rating'].fillna(0)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize KNN model
    knn = KNeighborsRegressor(n_neighbors=3, p=1)
    knn.fit(X_scaled, y)

    # Train TF-IDF Vectorizer for CBF
    cbf_features = ['neighbourhood_cleansed', 'room_type', 'property_type']
    data['combined_features'] = data[cbf_features].astype(str).apply(lambda x: ' '.join(x), axis=1)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data['combined_features'])

    # Create directories if they do not exist
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Save models
    with open(f'./models/knn_model_{city}.pkl', 'wb') as file:
        pickle.dump(knn, file)
    with open(f'./models/scaler_model_{city}.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    with open(f'./models/cbf_model_{city}.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

if __name__ == "__main__":
    cities = ['nyc', 'berlin', 'amsterdam', 'sydney', 'rome', 'tokyo', 'barcelona', 'brussels']
    for city in cities:
        print(f'Training models for {city}...')
        train_and_save_models(city)
        print(f'Models for {city} trained and saved.')

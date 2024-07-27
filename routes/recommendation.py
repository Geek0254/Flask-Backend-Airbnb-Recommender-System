from flask import Blueprint, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

recommendation_bp = Blueprint('recommendation', __name__)

# Load available cities from a file or hard-code them
available_cities = ['nyc', 'berlin', 'amsterdam', 'sydney', 'rome', 'tokyo', 'barcelona', 'brussels']

def load_data(city):
    file_path = f'./datasets/{city}/{city}_airbnb_listings.csv'
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data for {city}: {e}")
        return None

def load_model(city, model_type):
    try:
        with open(f'./models/{model_type}_model_{city}.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading {model_type} model for {city}: {e}")
        return None

def get_recommendations(data, user_preferences, features, vectorizer):
    user_pref_str = " ".join([str(user_preferences[feature]) for feature in features])
    data['combined_features'] = data[features].astype(str).apply(lambda x: ' '.join(x), axis=1)
    tfidf_matrix = vectorizer.transform(data['combined_features'])
    user_pref_tfidf = vectorizer.transform([user_pref_str])
    cosine_sim = cosine_similarity(user_pref_tfidf, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:10]]
    return data.iloc[top_indices]

@recommendation_bp.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        city = data['city']
        user_preferences = data['user_preferences']
        
        df = load_data(city)
        if df is None:
            return jsonify({"error": f"Data for city {city} could not be loaded."}), 500
        
        # Clean and convert price column
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Ensure data types are consistent
        df['number_of_reviews'] = df['number_of_reviews'].astype(int)
        df['availability_365'] = df['availability_365'].astype(int)
        df['review_scores_rating'] = df['review_scores_rating'].astype(float)
        df['minimum_nights'] = df['minimum_nights'].astype(int)
        df['beds'] = df['beds'].astype(float)
        df['bedrooms'] = df['bedrooms'].astype(float)

        filtered_data = df[
            (df['neighbourhood_cleansed'] == user_preferences['neighbourhood_cleansed']) &
            (df['room_type'] == user_preferences['room_type']) &
            (df['property_type'] == user_preferences['property_type']) &
            (df['price'] >= user_preferences['min_price']) &
            (df['price'] <= user_preferences['max_price']) &
            (df['number_of_reviews'] >= user_preferences['min_reviews']) &
            (df['availability_365'] >= user_preferences['min_availability']) &
            (df['review_scores_rating'] >= user_preferences['min_rating']) &
            (df['minimum_nights'] >= user_preferences['min_nights']) &
            (df['beds'] >= user_preferences['min_beds']) &
            (df['bedrooms'] >= user_preferences['min_bedrooms'])
        ]
        
        if filtered_data.empty:
            return jsonify({
                "message": "No listings match the given preferences using KNN. Do you want to proceed with Content-Based Filtering?",
                "status": "no_results_knn"
            }), 200
        else:
            knn_features = ['price', 'number_of_reviews', 'availability_365', 'minimum_nights', 'reviews_per_month', 'beds', 'bedrooms']
            X = filtered_data[knn_features].fillna(0)
            y = filtered_data['review_scores_rating'].fillna(0)
            scaler = load_model(city, 'scaler')
            knn = load_model(city, 'knn')
            if scaler is None or knn is None:
                return jsonify({"error": f"KNN model for city {city} could not be loaded."}), 500
            X_scaled = scaler.transform(X)
            filtered_data['Predicted Review Rate'] = knn.predict(X_scaled)
            recommendations = filtered_data.head(10)

        # Replace NaN with None for JSON serialization
        recommendations = recommendations.replace({np.nan: None})

        recommendations_table = recommendations[['name', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'price', 'number_of_reviews', 'availability_365', 'review_scores_rating', 'reviews_per_month', 'host_response_time', 'host_response_rate', 'instant_bookable', 'listing_url', 'picture_url']]
        return jsonify({
            "message": "Recommendations fetched successfully",
            "status": "results_found",
            "recommendations": recommendations_table.to_dict('records')
        }), 200

    except Exception as e:
        print(f"Error in recommendation endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/recommend_cbf', methods=['POST'])
def recommend_cbf():
    try:
        data = request.get_json()
        city = data['city']
        user_preferences = data['user_preferences']
        
        df = load_data(city)
        if df is None:
            return jsonify({"error": f"Data for city {city} could not be loaded."}), 500

        cbf_features = ['neighbourhood_cleansed', 'room_type', 'property_type']
        vectorizer = load_model(city, 'cbf')
        if vectorizer is None:
            return jsonify({"error": f"CBF model for city {city} could not be loaded."}), 500
        recommendations = get_recommendations(df, user_preferences, cbf_features, vectorizer)
        
        # Replace NaN with None for JSON serialization
        recommendations = recommendations.replace({np.nan: None})

        recommendations_table = recommendations[['name', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'price', 'number_of_reviews', 'availability_365', 'review_scores_rating', 'reviews_per_month', 'host_response_time', 'host_response_rate', 'instant_bookable', 'listing_url', 'picture_url']]
        return jsonify({
            "message": "Content-Based Filtering recommendations fetched successfully",
            "status": "cbf_results",
            "recommendations": recommendations_table.to_dict('records')
        }), 200

    except Exception as e:
        print(f"Error in recommend_cbf endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@recommendation_bp.route('/cities', methods=['GET'])
def get_cities():
    return jsonify({"cities": available_cities})

@recommendation_bp.route('/neighborhoods', methods=['GET'])
def get_neighborhoods():
    city = request.args.get('city')
    if not city or city not in available_cities:
        return jsonify({"error": "Invalid or missing city parameter"}), 400

    df = load_data(city)
    if df is None:
        return jsonify({"error": f"Data for city {city} could not be loaded."}), 500

    neighborhoods = df['neighbourhood_cleansed'].unique().tolist()
    return jsonify({"neighborhoods": neighborhoods})

@recommendation_bp.route('/room_types', methods=['GET'])
def get_room_types():
    city = request.args.get('city')
    if not city or city not in available_cities:
        return jsonify({"error": "Invalid or missing city parameter"}), 400

    df = load_data(city)
    if df is None:
        return jsonify({"error": f"Data for city {city} could not be loaded."}), 500

    room_types = df['room_type'].unique().tolist()
    return jsonify({"room_types": room_types})

@recommendation_bp.route('/property_types', methods=['GET'])
def get_property_types():
    city = request.args.get('city')
    if not city or city not in available_cities:
        return jsonify({"error": "Invalid or missing city parameter"}), 400

    df = load_data(city)
    if df is None:
        return jsonify({"error": f"Data for city {city} could not be loaded."}), 500

    property_types = df['property_type'].unique().tolist()
    return jsonify({"property_types": property_types})

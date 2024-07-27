from flask import Blueprint, request, jsonify
import pandas as pd
import pickle
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import io

recommendation_bp = Blueprint('recommendation', __name__)

# Pre-defined available cities
available_cities = ['nyc', 'berlin', 'amsterdam', 'sydney', 'rome', 'tokyo', 'barcelona', 'brussels']

# Google Drive file IDs for models
model_ids = {
    'cbf': {
        'amsterdam': '1ffV9JooTXfNRCCvoE4mwOSFam9HS9IRp',
        'barcelona': '1AVuEmY_tylkefageCOzpHzz91EoFe253',
        'berlin': '1obB0TMBirw0opgF9Og6Bzo3AOEpQaEQh',
        'brussels': '1a4bkOTda12J2_TDMwUkvqP6owMmC8Xdg',
        'nyc': '1UJ1zbdYXq-lxpQBik6N3xWvTpX0r_R6G',
        'rome': '1rx8HLOz94mxMLGzo976HGPr7-sdz13Mq',
        'sydney': '1FJpI3NUJwaeNpVDrSJAhJ976sNvZ5LS-',
        'tokyo': '1Wc1vakgSzEy0Yw8Kcum4edhGyL00J13T'
    },
    'knn': {
        'amsterdam': '1Yqra7lfjSycoA_FbIfv1WOZ0AekMk7rT',
        'barcelona': '1jSwC43RC2l3STLBaTuqgYKPnINKSaD7N',
        'berlin': '1wC_egbte22uQUWRJd6rmpEI44eBkV79U',
        'brussels': '1YasKDdScmBNG1I1zf0k1B0JLbX_BkYHL',
        'nyc': '1SRVERE12WuotFm7cT0LGpo58fAd61Giz',
        'rome': '1OTkwlhXrUigJaXQOY2ssA4b1Tzt4VF5n',
        'sydney': '1d2kldN4gCBv2goMIzmMg3tMegUKGtv2g',
        'tokyo': '1uumWSjJroNQZYAcKqMTnI0PoLvnshUt2'
    },
    'scaler': {
        'amsterdam': '19np-E4BUZ42aVwNoHVYmRVgJLPbfFuL5',
        'barcelona': '1L1PzCLWGD2tFaVVlr2d5eYvidnCUKcIO',
        'berlin': '1xHyAouE07BYrfIl3pEQxfGfBxKpm-Lw-',
        'brussels': '1_IirbDXJ_IU9l9XZLLeOihzhKPw5e4xP',
        'nyc': '12DZbgDcqXRrP6fZ7a5nIPHFlKZZk-Se-',
        'rome': '1EpXig7ivgVT-Wf4ZW9XzXhydyw2Ap9Bj',
        'sydney': '1-QKvBRLVj_ACBf4EyJ2HmJYBpPyeHffb',
        'tokyo': '1bDRoM2MqQV2ebq3Wmr2yGZER8PkSs0oo'
    },
    'data': {
        'amsterdam': '1MAB_koP-CPDkhjVs3DUbeU_jLuD3HCMD',
        'barcelona': '1vGDtFTscAXPHdtFHcbOBo47NW_qzEDSd',
        'berlin': '13MaITpBsID3lHpPadgDIMFJNNDs4nA5l',
        'brussels': '19EaSibkDr7VGdUd5Afo_JF1rMe_8MoEy',
        'nyc': '1U5LY2d6MPwVVjhECl5m8MpFpo-bDPLxj',
        'rome': '1ua6XGqqGCNq6yHBtFYxo6_DuH9sOfURD',
        'sydney': '1l0MY9e3FoV-qNkD9arAdjo8qIXvy4H0T',
        'tokyo': '1CI_JxuNs2ZY0rCHiw_YoE8SJg8lLME1w'
    }
}

def get_direct_download_link(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_file(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return response.content

def load_model(city, model_type):
    file_id = model_ids[model_type].get(city)
    if not file_id:
        print(f"No download link for {model_type} model for city: {city}")
        return None
    try:
        file_url = get_direct_download_link(file_id)
        content = download_file(file_url)
        model = pickle.load(io.BytesIO(content))
        return model
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

        df = load_model(city, 'data')  # Load the pre-processed dataframe model
        if df is None:
            return jsonify({"error": f"Data model for city {city} could not be loaded."}), 500

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

        df = load_model(city, 'data')  # Load the pre-processed dataframe model
        if df is None:
            return jsonify({"error": f"Data model for city {city} could not be loaded."}), 500

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

    df = load_model(city, 'data')
    if df is None:
        return jsonify({"error": f"Data for city {city} could not be loaded."}), 500

    neighborhoods = df['neighbourhood_cleansed'].unique().tolist()
    return jsonify({"neighborhoods": neighborhoods})

@recommendation_bp.route('/room_types', methods=['GET'])
def get_room_types():
    city = request.args.get('city')
    if not city or city not in available_cities:
        return jsonify({"error": "Invalid or missing city parameter"}), 400

    df = load_model(city, 'data')
    if df is None:
        return jsonify({"error": f"Data for city {city} could not be loaded."}), 500

    room_types = df['room_type'].unique().tolist()
    return jsonify({"room_types": room_types})

@recommendation_bp.route('/property_types', methods=['GET'])
def get_property_types():
    city = request.args.get('city')
    if not city or city not in available_cities:
        return jsonify({"error": "Invalid or missing city parameter"}), 400

    df = load_model(city, 'data')
    if df is None:
        return jsonify({"error": f"Data for city {city} could not be loaded."}), 500

    property_types = df['property_type'].unique().tolist()
    return jsonify({"property_types": property_types})

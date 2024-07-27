from flask import Flask, jsonify
from routes.recommendation import recommendation_bp

#Allow CORS for all domains on all routes
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.register_blueprint(recommendation_bp)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Airbnb Recommendation System API!"})

if __name__ == '__main__':
    app.run(debug=True)

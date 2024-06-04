from flask import Flask, jsonify
from routes.recommendation import recommendation_bp

app = Flask(__name__)
app.register_blueprint(recommendation_bp)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Airbnb Recommendation System API!"})

if __name__ == '__main__':
    app.run(debug=True)

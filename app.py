from flask import Flask, request, jsonify
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load preprocessed FAQs from JSON file
with open('preprocessed_faqs.json', 'r') as f:
    preprocessed_faqs = json.load(f)

# Vectorize the preprocessed FAQ questions
faq_vectors = [nlp(faq['question']).vector for faq in preprocessed_faqs]

# Function to get the most similar FAQ answer
def get_most_similar_faq(user_query):
    # Preprocess and vectorize the user query
    user_query_vector = nlp(user_query).vector
    # Calculate cosine similarities
    similarities = cosine_similarity([user_query_vector], faq_vectors)
    # Find the index of the most similar FAQ
    most_similar_index = np.argmax(similarities)
    # Return the answer of the most similar FAQ
    return preprocessed_faqs[most_similar_index]['answer']

# Initialize Flask application
app = Flask(__name__)

# Define a route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_query = request.json.get('query')
    response = get_most_similar_faq(user_query)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

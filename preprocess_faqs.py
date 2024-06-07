import spacy
import json

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Example FAQs
faqs = [
    {"question": "What are your opening hours?", "answer": "Our branches are open from 9 AM to 5 PM, Monday to Friday."},
    {"question": "How can I open a bank account?", "answer": "You can open a bank account by visiting any of our branches with a valid ID and proof of address."},
    {"question": "What is the interest rate for savings accounts?", "answer": "The current interest rate for savings accounts is 1.5% per annum."},
    {"question": "How do I apply for a loan?", "answer": "You can apply for a loan online through our website or by visiting a branch."},
    {"question": "What documents are required for a mortgage application?", "answer": "You need to provide proof of income, credit history, and a valid ID."},
    {"question": "How can I reset my online banking password?", "answer": "You can reset your password by clicking 'Forgot Password' on the login page."},
    {"question": "What are the fees for international wire transfers?", "answer": "The fees for international wire transfers vary depending on the destination and amount. Please check our website for detailed information."},
    {"question": "Can I access my account from abroad?", "answer": "Yes, you can access your account from anywhere using our online banking service."},
    {"question": "How do I report a lost or stolen card?", "answer": "Report a lost or stolen card immediately by calling our 24/7 customer service line."},
    {"question": "What is the minimum balance requirement for a checking account?", "answer": "The minimum balance requirement for a checking account is $100."}
]

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

# Preprocess all questions
preprocessed_faqs = [{'question': preprocess_text(faq['question']), 'answer': faq['answer']} for faq in faqs]

# Save preprocessed FAQs to a JSON file
with open('preprocessed_faqs.json', 'w') as f:
    json.dump(preprocessed_faqs, f, indent=4)

print("Preprocessing complete. Preprocessed FAQs saved to 'preprocessed_faqs.json'.")

from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import nltk
import google.generativeai as genai

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set up DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set up Google Gemini API
genai.configure(api_key='AIzaSyDuUvX_FlvDVveb90NLSga3eEqhnrCdZWA')  # Replace with your actual Gemini API key
gemini_model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)

def analyze_with_distilbert(question, keywords, response):
    combined_input = f"{question} {' '.join(keywords)} {response}"
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment_score = probabilities[0][1].item() * 100  # Convert to percentage
    
    # Calculate relevance score based on keyword matching and response length
    keyword_match = sum(keyword.lower() in response.lower() for keyword in keywords)
    keyword_relevance = (keyword_match / len(keywords)) * 50  # 50% weight to keyword matching
    length_relevance = min(len(response.split()) / 50, 1) * 50  # 50% weight to response length, capped at 50 words
    relevance_score = keyword_relevance + length_relevance

    # Adjust sentiment score based on relevance
    sentiment_score = (sentiment_score + relevance_score) / 2

    return sentiment_score, relevance_score

def clean_feedback(feedback_text):
    """
    Cleans up the feedback text to remove unwanted characters like asterisks (*)
    and ensures proper formatting.
    """
    # Remove asterisks and extra whitespace
    cleaned_feedback = feedback_text.replace('*', '').strip()
    
    # Replace multiple spaces or newlines with clean paragraph formatting
    cleaned_feedback = cleaned_feedback.replace('\n\n', '\n').replace('\n ', '\n')
    
    return cleaned_feedback

def get_feedback_from_gemini(question, keywords, response, sentiment_score, relevance_score):
    prompt = f"""
    Question: {question}
    Keywords: {', '.join(keywords)}
    Student Response: {response}
    Sentiment Score: {sentiment_score:.2f}%
    Relevance Score: {relevance_score:.2f}%

    Based on the above information, provide a detailed feedback on the student's response.
    Consider the following points:
    1. How well does the response answer the question?
    2. Does the response cover the key concepts (keywords)?
    3. What are the strengths and weaknesses of the response?
    4. Suggestions for improvement.

    Format your feedback using proper paragraphs and bullet points for readability.
    Provide your feedback in a clear and constructive manner.
    Do not use asterisks (*) in your response.

    After providing the feedback, include a section titled "Model Answer" with a comprehensive example answer that would score 100% in relevance and sentiment.
    """

    generated_response = gemini_model.generate_content(prompt)
    feedback = clean_feedback(generated_response.text)

    return feedback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    question = data['question']
    keywords = [k.strip().lower() for k in data['keywords'].split(',')]
    response = data['response']
    
    # Analyze the response using DistilBERT
    sentiment_score, relevance_score = analyze_with_distilbert(question, keywords, response)
    
    # Get feedback from the Gemini model and ensure it's clean (without asterisks)
    feedback = get_feedback_from_gemini(question, keywords, response, sentiment_score, relevance_score)
    
    # Check which keywords were addressed
    keywords_addressed = [keyword for keyword in keywords if keyword in response.lower()]
    
    analysis = {
        "sentiment_score": sentiment_score,
        "relevance_score": relevance_score,
        "keywords_addressed": keywords_addressed,
        "feedback": feedback
    }
    
    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)

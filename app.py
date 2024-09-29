from flask import Flask, request, jsonify
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# Initialize Flask app
app = Flask(__name__)

# Load Spacy model for NER
nlp = spacy.load('en_core_web_sm')

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def evaluate_faithfulness(data_input, llm_response):
    # Extract entities
    data_entities = set(extract_entities(data_input))
    response_entities = set(extract_entities(llm_response))
    
    # Calculate precision, recall, and F1 score for entities
    true_positives = data_entities & response_entities
    false_positives = response_entities - data_entities
    false_negatives = data_entities - response_entities
    
    precision = len(true_positives) / (len(true_positives) + len(false_positives) + 1e-10)
    recall = len(true_positives) / (len(true_positives) + len(false_negatives) + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': list(true_positives),
        'false_positives': list(false_positives),
        'false_negatives': list(false_negatives)
    }

def evaluate_relevance(user_prompt, llm_response):
    # Get embeddings
    embeddings = model.encode([user_prompt, llm_response])
    cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    return {
        'semantic_similarity': cos_sim
    }

def evaluate_context_utilization(data_input, llm_response):
    # Calculate ROUGE scores
    scores = scorer.score(data_input, llm_response)
    
    rouge1_fmeasure = scores['rouge1'].fmeasure
    rougeL_fmeasure = scores['rougeL'].fmeasure
    
    return {
        'rouge1_f1_score': rouge1_fmeasure,
        'rougeL_f1_score': rougeL_fmeasure
    }

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    user_prompt = data.get('user_prompt', '')
    data_input = data.get('data_input', '')
    llm_response = data.get('llm_response', '')
    
    if not user_prompt or not data_input or not llm_response:
        return jsonify({'error': 'user_prompt, data_input, and llm_response are required'}), 400
    
    # Perform evaluations
    faithfulness_scores = evaluate_faithfulness(data_input, llm_response)
    relevance_scores = evaluate_relevance(user_prompt, llm_response)
    context_scores = evaluate_context_utilization(data_input, llm_response)
    
    # Combine results
    result = {
        'faithfulness': faithfulness_scores,
        'relevance': relevance_scores,
        'context_utilization': context_scores
    }
    
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

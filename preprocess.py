import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import re
from nltk import download, word_tokenize, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean
from vncorenlp import VnCoreNLP



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vncorenlp = VnCoreNLP(r"C:\Users\nguye\Downloads\VnCoreNLP-master\VnCoreNLP-master\VnCoreNLP-1.2.jar", 
                      annotators="wseg,pos,ner", 
                      port=62328)


download('punkt')

def load_stopwords(directory):
    stopwords = set()
    for filename in ['vietnamese-stopwords.txt', 'vietnamese-stopwords-dash.txt']:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                stopwords.update(line.strip().replace('_', ' ') for line in f)
        else:
            print(f"File not found: {file_path}")
    return stopwords

def load_synonyms_from_files(directory):
    synonyms_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split(',')
                    base_word = words[0].strip()
                    synonyms = set(word.strip() for word in words[1:])
                    synonyms_dict.setdefault(base_word, set()).update(synonyms)
    return synonyms_dict

def label_to_id(label):
    if isinstance(label, (int, float)):
        return int(label)
    elif isinstance(label, str):
        return 1 if label.lower() in ['true', 'yes'] else 0
    else:
        return 0

def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

def convert_ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, np.int32):
        return int(data)
    elif isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    else:
        return data

def save_json(data, file_path):
    data = convert_ndarray_to_list(data)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base").to(device)

def get_phobert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def compute_euclidean_distance(vec1, vec2):
    try:
        dist = euclidean(vec1, vec2)
        return 1 / (1 + dist)  
    except ZeroDivisionError:
        return 1.0

def preprocess_and_normalize(text):
    text = text.lower()
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '<DATE>', text)
    tokens = vncorenlp.tokenize(text)[0]
    return ' '.join(tokens)


def preprocess_text_combined(text, synonyms_dict, stopwords, tfidf_vectorizer, max_length=1000):
    def split_text(text, max_length):
        sentences = text.split('. ')
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += ' ' + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def preprocess_chunk(chunk, synonyms_dict, stopwords, tfidf_vectorizer):
        words = [str(synonyms_dict.get(word, word)) for word in tokens if word.lower() not in stopwords]
        
        processed_chunk = ' '.join(words)
        
        tfidf_vector = tfidf_vectorizer.transform([processed_chunk]).toarray()
        
        return processed_chunk, tfidf_vector

    text_chunks = split_text(text, max_length)
    all_named_entities = []
    all_tokens = []

    for chunk in text_chunks:
        try:
            named_entities = vncorenlp.ner(chunk)[0]
            if named_entities:
                print(f"Named Entities in chunk: {named_entities}")
        except Exception as e:
            print(f"Error during NER processing: {e}")
            named_entities = []
        
        all_named_entities.extend(named_entities)
        
        processed_chunk, tfidf_vector = preprocess_chunk(chunk, synonyms_dict, stopwords, tfidf_vectorizer)
        
        all_tokens.extend(processed_chunk.split())

    embeddings = get_phobert_embeddings(text)
    
    return {
        'tokens': all_tokens,
        'embeddings': embeddings,
        'tfidf': tfidf_vector
    }, all_named_entities




def preprocess_data(data, legal_passages, synonyms_dict, stopwords):
    legal_dict = {law['id']: {article['id']: article['text']
                              for article in law.get('articles', [])} 
                  for law in legal_passages}

    all_texts = []
    for item in data:
        all_texts.append(item.get('statement', ''))
        for passage in item.get('legal_passages', []):
            law_id = passage.get('law_id')
            article_id = passage.get('article_id')
            if law_id in legal_dict and article_id in legal_dict[law_id]:
                passage_text = legal_dict[law_id][article_id]
                all_texts.append(passage_text)

    tfidf_vectorizer = TfidfVectorizer().fit(all_texts)
    
    processed_data = [preprocess_item(item, legal_dict, synonyms_dict, stopwords, tfidf_vectorizer) 
                      for item in data]
    
    return processed_data


def preprocess_item(item, legal_dict, synonyms_dict, stopwords, tfidf_vectorizer):
    statement_text = item.get('statement', '')
    processed_statement, _ = preprocess_text_combined(statement_text, synonyms_dict, stopwords, tfidf_vectorizer)

    legal_passages_embeddings = []
    legal_passages_tfidf = []
    for passage in item.get('legal_passages', []):
        law_id = passage.get('law_id')
        article_id = passage.get('article_id')
        if law_id in legal_dict and article_id in legal_dict[law_id]:
            passage_text = legal_dict[law_id][article_id]
            processed_passage, _ = preprocess_text_combined(passage_text, synonyms_dict, stopwords, tfidf_vectorizer)
            legal_passages_embeddings.append(processed_passage['embeddings'])
            legal_passages_tfidf.append(processed_passage['tfidf'])

    statement_features = processed_statement['embeddings']
    statement_tfidf = processed_statement['tfidf']

    similarities = []
    for passage_features, passage_tfidf in zip(legal_passages_embeddings, legal_passages_tfidf):
        cosine_sim = compute_cosine_similarity(statement_features, passage_features)
        euclidean_dist = compute_euclidean_distance(statement_features, passage_features)
        similarities.append((cosine_sim, euclidean_dist, tfidf_cosine_sim))

    max_cosine_sim = max(sim[0] for sim in similarities) if similarities else 0
    max_euclidean_dist = max(sim[1] for sim in similarities) if similarities else 0
    max_tfidf_cosine_sim = max(sim[2] for sim in similarities) if similarities else 0

    threshold_cosine = 0.8
    threshold_euclidean = 0.3
    threshold_tfidf = 0.7

    predicted_label = 'yes' if (max_cosine_sim >= threshold_cosine or 
                                 max_euclidean_dist >= threshold_euclidean or
                                 max_tfidf_cosine_sim >= threshold_tfidf) else 'no'

    return {
        'example_id': item.get('example_id', ''),
        'statement': statement_text,
        'tokens': processed_statement['tokens'],
        'original_label': item.get('label', 'Unknown'),
        'predicted_label': predicted_label,
        'legal_passages': legal_passages_embeddings,
        'similarity': {
            'cosine': max_cosine_sim,
            'euclidean': max_euclidean_dist,
            'tfidf_cosine': max_tfidf_cosine_sim
        }
    }


def main():
    stopwords_directory = 'vi-stopword'
    synonyms_directory = 'vi-wordnet'
    
    stopwords = load_stopwords(stopwords_directory)
    synonyms_dict = load_synonyms_from_files(synonyms_directory)
    
    train_file = 'train.json'
    test_file = 'test.json'
    legal_passages_file = 'legal_passages.json'
    
    train_data = load_json(train_file)
    test_data = load_json(test_file)
    legal_passages = load_json(legal_passages_file)
    
    if train_data and legal_passages:
        processed_train_data = preprocess_data(train_data, legal_passages, synonyms_dict, stopwords)
        save_json(processed_train_data, 'processed_train.json')
    
    if test_data and legal_passages:
        processed_test_data = preprocess_data(test_data, legal_passages, synonyms_dict, stopwords)
        save_json(processed_test_data, 'processed_test.json')

if __name__ == "__main__":
    main()
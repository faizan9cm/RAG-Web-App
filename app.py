import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Avoid conflicting libraries

import anthropic
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import torch
import requests
import re


app = Flask(__name__)


# Global variables for storing Faiss index and chunks
faiss_index = None
chunks = None


# Scrape the Wikipedia page
def scrape_wikipedia_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', {'id': 'bodyContent'})
    if content_div:
        content = content_div.get_text(separator='\n')
        return content
    else:
        return "Content not found"

# Clean pattern
def clean_text(text):
    cleaned_text = re.sub(r'\(Episode [IVX]+\)', '', text)
    return cleaned_text


# Chunk the text content
def chunk_text(text, chunk_size=300):
    cleaned_text = clean_text(text)
    sentences = cleaned_text.split('\n')
    chunks = []
    chunk = ''
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + '\n'
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
    if chunk:
        chunks.append(chunk.strip())
    return chunks


# Generate embeddings and store in Faiss
def embed_chunks(chunks, model_name ='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            model_output = model(**inputs)
        chunk_embedding = mean_pooling(model_output, inputs['attention_mask'])
        embeddings.append(chunk_embedding[0].numpy())
    return np.vstack(embeddings)

def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Implement functionality to query the Faiss index
def find_relevant_chunks(question, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3):
    global faiss_index, chunks
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**inputs)
    question_embedding = mean_pooling(model_output, inputs['attention_mask'])[0].numpy()

    distances, indices = faiss_index.search(np.array([question_embedding]), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


def get_answer(question):
    global faiss_index, chunks
    relevant_chunks = find_relevant_chunks(question)
    context = "\n".join(relevant_chunks)
    
    # Call the Claude API
    client = anthropic.Anthropic(
        api_key="api-key",
    )
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="You are an AI assistant.",
        messages=[{"role": "user", "content": question}]
    )
    completion_text = ''.join([block.text for block in message.content])
    return completion_text.split('.')  # Split answer into sentences


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    global faiss_index, chunks
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = get_answer(question)
    
    return jsonify({"answer": answer})


if __name__ == '__main__':    
    # Preload data
    url = 'https://en.wikipedia.org/wiki/Luke_Skywalker'
    content = scrape_wikipedia_page(url)
    chunks = chunk_text(content)
    embeddings = embed_chunks(chunks)
    faiss_index = store_embeddings_in_faiss(embeddings)
    
    app.run(debug=True)

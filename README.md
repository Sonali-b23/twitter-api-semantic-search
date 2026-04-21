ML Tasks: Semantic Search & News Narrative Builder

Overview

This repository contains implementations of two machine learning systems:

1. Semantic Search Engine over Twitter API documentation
2. News Narrative Builder for generating structured storylines from news datasets

Both tasks leverage modern NLP techniques such as Sentence-BERT embeddings, semantic similarity, and clustering.

---

Project Structure

├── twitter-api-semantic-search/
│   ├── semantic_search.py
│   ├── requirements.txt
│   └── data/
│
├── news-narrative-builder/
│   ├── narrative_builder.py
│   ├── requirements.txt
│   └── news_dataset.json
│
└── README.md

---

Installation

Clone the repository:

git clone https://github.com/Sonali-b23/twitter-api-semantic-search.git
cd twitter-api-semantic-search

Install dependencies:

pip install -r requirements.txt

---

Task 1: Semantic Search on Twitter API Documentation

Objective

Build a semantic search system that retrieves the most relevant Twitter API endpoints based on a natural language query.

Approach

1. Data Preparation
   
   - Extract documentation text from Postman collection
   - Split into meaningful chunks

2. Embedding Generation
   
   - Use Sentence-BERT to convert text into dense vectors

3. Indexing
   
   - Store embeddings using FAISS for fast similarity search

4. Query Processing
   
   - Convert user query into embedding
   - Retrieve top-k similar chunks using cosine similarity

---

Features

- Natural language query support
- High-speed retrieval using FAISS
- Context-aware results using transformer embeddings

---

Usage

Run the semantic search:

python semantic_search.py --query "How do I fetch tweets with expansions?"

Example:

python semantic_search.py --query "How do I filter tweets by date?"

---

Sample Output

Top Results:
1. Recent Search Endpoint - Retrieves tweets filtered by time range
2. Multiple Tweets Endpoint - Supports expansions and fields

---

Results

The system successfully retrieves the most relevant API endpoints by mapping query intent to documentation semantics using vector similarity.

---

Task 2: News Narrative Builder

Objective

Generate a structured narrative from a large news dataset based on a given topic.

---

Pipeline

1. Filtering
   
   - Select articles relevant to the input topic using semantic similarity

2. Embedding
   
   - Convert articles into vector representations

3. Clustering
   
   - Group similar articles using clustering (e.g., K-Means / hierarchical clustering)

4. Timeline Construction
   
   - Sort articles chronologically

5. Narrative Generation
   
   - Generate a 5–10 sentence summary of the overall storyline

6. Graph Construction
   
   - Build relationships between articles:
     - "builds on"
     - "contradicts"
     - "related to"

---

Features

- Narrative summary generation
- Chronological timeline
- Semantic clustering of articles
- Graph-based relationship mapping

---

Usage

Run the narrative builder:

python news-narrative-builder/narrative_builder.py --topic "Jubilee Hills elections"

Example:

python news-narrative-builder/narrative_builder.py --topic "Israel-Iran conflict"

---

Output Format

The system generates a JSON file with the following structure:

{
  "narrative_summary": "...",
  "timeline": [...],
  "clusters": [...],
  "graph": {...}
}

---

Results

The model successfully produces:

- A coherent narrative summary
- A chronological timeline of events
- 5 semantic clusters of related articles
- A graph representing inter-article relationships

---

Future Improvements

- Add evaluation metrics (precision@k for search)
- Improve clustering using advanced methods (HDBSCAN)
- Visualize narrative graph using interactive tools
- Deploy as a web application

---

Tech Stack

- Python
- Sentence-BERT
- FAISS
- NumPy / Pandas
- Scikit-learn

---

Conclusion

This project demonstrates practical applications of NLP in:

- Information retrieval (semantic search)
- Story understanding (narrative generation)

It highlights how embeddings and similarity learning can be used to build intelligent, real-world systems.

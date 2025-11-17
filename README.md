# ML Tasks - Semantic Search and Narrative Builder

This repository contains solutions for two core machine learning tasks:

1. **Semantic Search on Twitter API Documentation**  
2. **News Narrative Builder**

Both tasks are implemented in separate Python files but reside within the same repository.

**Task 1: Semantic Search on Twitter API Documentation**

Objective:
Build a **semantic search engine** over the **Twitter API Postman documentation**. The system performs semantic search on the documentation and returns the most relevant chunks for a given query.

Features:
- **Semantic Search**: Query the Twitter API documentation with a natural language query.
- **FAISS Indexing**: Index documentation chunks using FAISS for fast retrieval.
- **Sentence-BERT Embedding**: Use pre-trained embeddings for efficient semantic search.

Setup Instructions:

 1. Clone the Repository:
git clone https://github.com/Sonali-b23/twitter-api-semantic-search.git

3. Install Dependencies:
pip install -r twitter-api-semantic-search/requirements.txt

3. Run the Semantic Search Script:
You can query the documentation using the following command:
python twitter-api-semantic-search/semantic_search.py --query "How do I fetch tweets with expansions?

Example Query:
python twitter-api-semantic-search/semantic_search.py --query "How do I filter tweets by date?"

**Task 2: News Narrative Builder***
Objective:
This task processes a large JSON news dataset, filters the relevant articles based on a user-provided topic, and generates a narrative. The narrative includes:

A summary of the topic.
A timeline of events.
Clusters of semantically related articles.
A narrative graph representing relationships between articles.

Features:
**Narrative Summary**: A 5–10 sentence synthesis of the main storyline around a topic.
**Timeline of Events**: A chronological list of relevant articles related to the topic.
**Clustering:** Articles are grouped into semantic clusters.
**Narrative Graph**: Visualizes the relationships between articles (e.g., "builds on", "contradicts").

**Setup Instructions:**
1. Clone the Repository:
git clone https://github.com/Sonali-b23/twitter-api-semantic-search.git

2. Install Dependencies:
pip install -r news-narrative-builder/requirements.txt

3. Run the Narrative Builder Script:
To generate a narrative for a topic, use the following command:
python news-narrative-builder/narrative_builder.py --topic "Jubilee Hills elections"

**This will generate:**
A narrative summary
A timeline of events.
Clusters of related articles.
A narrative graph of article relationships.

**Example Command:**

python news-narrative-builder/narrative_builder.py --topic "Israel-Iran conflict"


**Output Structure:**
The output will be a JSON file containing:
narrative_summary: A short summary of the main storyline.
timeline: A chronological ordering of relevant articles.
clusters: Groups of semantically similar articles.
graph: A graph showing relationships between articles.

**news_dataset.json: JSON file containing the news articles dataset.**


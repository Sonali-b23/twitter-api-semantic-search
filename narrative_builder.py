import argparse
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import numpy as np

# -----------------------------
# Load and filter dataset
# -----------------------------
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check for top-level "items" key
    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    # Filter by source_rating > 8
    filtered = [article for article in data if isinstance(article, dict) and article.get("source_rating", 0) > 8]

    if not filtered:
        print("Warning: No articles found with source_rating > 8. Using all articles instead.")
        filtered = [article for article in data if isinstance(article, dict)]

    return filtered

# -----------------------------
# Get relevant articles for topic
# -----------------------------
def retrieve_relevant_articles(articles, topic, model, top_k=50):
    texts = [a["title"] + " " + a.get("story", "") for a in articles]  # Change 'headline' to 'title'
    embeddings = model.encode(texts, convert_to_numpy=True)
    topic_emb = model.encode([topic], convert_to_numpy=True)
    sims = cosine_similarity(topic_emb, embeddings)[0]
    
    top_indices = np.argsort(sims)[-top_k:][::-1]
    relevant = [articles[i] for i in top_indices]
    return relevant, embeddings[top_indices]

# -----------------------------
# Generate narrative summary
# -----------------------------
def generate_summary(articles, max_sentences=10):
    titles = [a["title"] for a in articles]  
    summary = " ".join(titles[:max_sentences])
    return summary

# -----------------------------
# Build timeline
# -----------------------------
def build_timeline(articles):
    timeline = []
    for a in sorted(articles, key=lambda x: x["published_at"]):  # Use 'published_at' for sorting
        timeline.append({
            "date": a["published_at"],
            "headline": a["title"],  
            "url": a.get("url", ""),
            "why_it_matters": a.get("story", "")
        })
    return timeline

# -----------------------------
# Build clusters
# -----------------------------
def build_clusters(embeddings, articles, n_clusters=5):
    clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(articles)))
    labels = clustering.fit_predict(embeddings)
    clusters = []
    for i in range(max(labels)+1):
        cluster_articles = [articles[j] for j in range(len(articles)) if labels[j] == i]
        clusters.append({
            "cluster_id": i,
            "articles": [{"title": a["title"], "url": a.get("url", "")} for a in cluster_articles]  # Changed 'headline' to 'title'
        })
    return clusters

# -----------------------------
# Build narrative graph
# -----------------------------
def build_graph(articles, embeddings):
    G = nx.Graph()
    for idx, a in enumerate(articles):
        G.add_node(idx, title=a.get("title", ""), url=a.get("url", ""))
    
    sims = cosine_similarity(embeddings)
    for i in range(len(articles)):
        for j in range(i+1, len(articles)):
            if sims[i, j] > 0.7:  # threshold for "related"
                G.add_edge(i, j, relation="related")
    
    graph_data = nx.readwrite.json_graph.node_link_data(G)
    return graph_data

# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Topic to generate narrative for")
    parser.add_argument("--dataset", default="news_dataset.json", help="Path to news dataset JSON")
    args = parser.parse_args()

    print("Loading dataset...")
    articles = load_dataset(args.dataset)
    if not articles:
        print("No articles found in dataset. Exiting.")
        return
    print(f"{len(articles)} articles loaded.")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Retrieving relevant articles...")
    relevant_articles, embeddings = retrieve_relevant_articles(articles, args.topic, model)

    print("Generating narrative summary...")
    summary = generate_summary(relevant_articles)

    print("Building timeline...")
    timeline = build_timeline(relevant_articles)

    print("Building clusters...")
    clusters = build_clusters(embeddings, relevant_articles)

    print("Building narrative graph...")
    graph = build_graph(relevant_articles, embeddings)

    output = {
        "narrative_summary": summary,
        "timeline": timeline,
        "clusters": clusters,
        "graph": graph
    }

    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()

import argparse
import json
from datetime import datetime

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load and filter dataset
# -----------------------------
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data:
        data = data["items"]

    filtered = [a for a in data if isinstance(a, dict) and a.get("source_rating", 0) > 8]

    if not filtered:
        print("Warning: no articles found with source_rating > 8. Using all articles instead.")
        filtered = [a for a in data if isinstance(a, dict)]

    return filtered


# -----------------------------
# Get relevant articles for topic
# -----------------------------
def retrieve_relevant_articles(articles, topic, model, top_k=50):
    texts = [a.get("title", "") + " " + a.get("story", "") for a in articles]
    embeddings = model.encode(texts, convert_to_numpy=True)
    topic_emb = model.encode([topic], convert_to_numpy=True)
    sims = cosine_similarity(topic_emb, embeddings)[0]

    top_k = min(top_k, len(articles))
    top_indices = np.argsort(sims)[-top_k:][::-1]
    relevant = [articles[i] for i in top_indices]
    return relevant, embeddings[top_indices]


# -----------------------------
# Generate narrative summary
# -----------------------------
_SUMMARIZER = None


def _get_summarizer():
    """
    Lazily loads a small instruction-tuned model (flan-t5-small, ~80M params,
    CPU-friendly) to write an actual narrative paragraph, instead of just
    concatenating headlines. Falls back gracefully (with a warning) if
    `transformers`/`torch` aren't installed or the model can't be downloaded,
    so the rest of the pipeline still works without this dependency.
    """
    global _SUMMARIZER
    if _SUMMARIZER is not None:
        return _SUMMARIZER
    try:
        from transformers import pipeline
        _SUMMARIZER = pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception as e:
        print(f"Warning: could not load narrative-generation model ({e}). "
              f"Falling back to a plain headline join. Install `transformers` and `torch`, "
              f"and ensure you have network access to Hugging Face, to enable real narrative generation.")
        _SUMMARIZER = False
    return _SUMMARIZER


def generate_summary(topic, articles, timeline, max_input_articles=15):
    """
    Generates a short narrative paragraph describing how the story around
    `topic` unfolds, using the chronological timeline. Uses an actual
    text-generation model when available; otherwise falls back to a plain
    join of headlines (the original behavior) so the function never fails.
    """
    ordered_titles = [t["headline"] for t in timeline[:max_input_articles] if t.get("headline")]
    if not ordered_titles:
        return ""

    summarizer = _get_summarizer()
    if not summarizer:
        return " ".join(ordered_titles[:10])

    prompt = (
        f"Write a short, 4-6 sentence narrative summary describing how the story about "
        f"'{topic}' developed over time, based on these headlines in chronological order:\n"
        + "\n".join(f"- {t}" for t in ordered_titles)
    )
    try:
        output = summarizer(prompt, max_new_tokens=200, do_sample=False)
        return output[0]["generated_text"].strip()
    except Exception as e:
        print(f"Warning: narrative generation failed at inference time ({e}). Falling back to headline join.")
        return " ".join(ordered_titles[:10])


# -----------------------------
# Build timeline
# -----------------------------
def build_timeline(articles):
    timeline = []
    for a in sorted(articles, key=lambda x: x.get("published_at", "")):
        timeline.append({
            "date": a.get("published_at", ""),
            "headline": a.get("title", ""),
            "url": a.get("url", ""),
            "why_it_matters": a.get("story", "")[:300],
        })
    return timeline


# -----------------------------
# Build clusters
# -----------------------------
def build_clusters(embeddings, articles, n_clusters=5):
    n_clusters = min(n_clusters, len(articles))
    if n_clusters < 2:
        return [{"cluster_id": 0, "articles": [{"title": a.get("title", ""), "url": a.get("url", "")} for a in articles]}]

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(embeddings)
    clusters = []
    for i in range(max(labels) + 1):
        cluster_articles = [articles[j] for j in range(len(articles)) if labels[j] == i]
        clusters.append({
            "cluster_id": i,
            "articles": [{"title": a.get("title", ""), "url": a.get("url", "")} for a in cluster_articles],
        })
    return clusters


# -----------------------------
# Build narrative graph
# -----------------------------
_NLI_MODEL = None
RELATED_THRESHOLD = 0.7
CONTRADICTION_THRESHOLD = 0.6  # only meaningful among already-"related" pairs


def _get_nli_model():
    """
    Lazily loads a small cross-encoder NLI model to distinguish "contradicts"
    from plain "related" among topically-similar article pairs. This is a
    genuine (if imperfect) classifier, not a hardcoded label -- but it's only
    run on pairs that already passed the similarity threshold, since two
    unrelated articles can't meaningfully "contradict" each other.
    """
    global _NLI_MODEL
    if _NLI_MODEL is not None:
        return _NLI_MODEL
    try:
        from sentence_transformers import CrossEncoder
        _NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    except Exception as e:
        print(f"Warning: could not load contradiction-detection model ({e}). "
              f"All related pairs will be labeled 'related' rather than distinguishing 'contradicts'.")
        _NLI_MODEL = False
    return _NLI_MODEL


def build_graph(articles, embeddings):
    G = nx.Graph()
    for idx, a in enumerate(articles):
        G.add_node(idx, title=a.get("title", ""), url=a.get("url", ""))

    sims = cosine_similarity(embeddings)
    nli_model = _get_nli_model()

    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            if sims[i, j] <= RELATED_THRESHOLD:
                continue

            relation = "related"
            if nli_model:
                text_i = (articles[i].get("title", "") + ". " + articles[i].get("story", ""))[:512]
                text_j = (articles[j].get("title", "") + ". " + articles[j].get("story", ""))[:512]
                try:
                    scores = nli_model.predict([(text_i, text_j)])[0]
                    # cross-encoder/nli-deberta-v3-small label order: [contradiction, entailment, neutral]
                    if scores[0] == max(scores):
                        relation = "contradicts"
                except Exception:
                    pass  # keep default "related" label on any per-pair inference failure

            edge_attrs = {"relation": relation}
            if relation == "related":
                date_i, date_j = articles[i].get("published_at", ""), articles[j].get("published_at", "")
                # "builds_on" = a related article published later, but only within a plausible
                # follow-up window (90 days) -- two related articles years apart are still just
                # "related", not one directly continuing the other's story.
                if date_i and date_j and date_i != date_j:
                    try:
                        d_i = datetime.fromisoformat(date_i.replace("Z", "+00:00"))
                        d_j = datetime.fromisoformat(date_j.replace("Z", "+00:00"))
                        if abs((d_j - d_i).days) <= 90:
                            edge_attrs["relation"] = "builds_on"
                            edge_attrs["earlier"] = i if d_i < d_j else j
                            edge_attrs["later"] = j if d_i < d_j else i
                    except (ValueError, TypeError):
                        pass  # unparseable dates -> leave as "related"

            G.add_edge(i, j, **edge_attrs)

    return nx.readwrite.json_graph.node_link_data(G, edges="links")


# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Topic to generate narrative for")
    parser.add_argument("--dataset", default="news_dataset_sample.json", help="Path to news dataset JSON")
    parser.add_argument("--top_k", type=int, default=50, help="Number of relevant articles to retrieve")
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="If set, write the JSON result to this file instead of only printing it")
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
    relevant_articles, embeddings = retrieve_relevant_articles(articles, args.topic, model, top_k=args.top_k)

    print("Building timeline...")
    timeline = build_timeline(relevant_articles)

    print("Generating narrative summary...")
    summary = generate_summary(args.topic, relevant_articles, timeline)

    print("Building clusters...")
    clusters = build_clusters(embeddings, relevant_articles, n_clusters=args.n_clusters)

    print("Building narrative graph (related / builds_on / contradicts)...")
    graph = build_graph(relevant_articles, embeddings)

    output = {
        "topic": args.topic,
        "narrative_summary": summary,
        "timeline": timeline,
        "clusters": clusters,
        "graph": graph,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved result to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

import argparse
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Step 1: Load and chunk documentation
# -----------------------------
def load_postman_docs(repo_path="postman-twitter-api"):
    """
    Loads all Postman JSON files recursively and extracts meaningful chunks.
    """
    chunks = []

    def extract_items(items):
        for item in items:
            # If the item has a "request", it's an endpoint
            if "request" in item:
                name = item.get("name", "")
                request = item.get("request", {})
                desc = request.get("description", "")

                # Sometimes description is a dict with "content"
                if isinstance(desc, dict):
                    desc = desc.get("content", "")

                if desc:
                    chunks.append({
                        "name": name,
                        "description": desc,
                        "raw": item
                    })
            # If the item has nested items, recurse
            if "item" in item:
                extract_items(item["item"])

    for root, dirs, files in os.walk(repo_path):
        for f in files:
            if f.endswith(".json"):
                file_path = os.path.join(root, f)
                try:
                    with open(file_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    continue

                if "item" in data:
                    extract_items(data["item"])

    return chunks


# -----------------------------
# Step 2: Build embeddings
# -----------------------------
def embed_chunks(chunks, model):
    texts = [c["description"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

# -----------------------------
# Step 3: Create FAISS index
# -----------------------------
def build_faiss_index(embeddings):
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    return idx

# -----------------------------
# Step 4: Query search engine
# -----------------------------
def query_docs(query, model, index, chunks, top_k=5):
    q_emb = model.encode([query]).astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    D, I = index.search(q_emb, top_k)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        item = chunks[idx]
        results.append({
            "rank": rank + 1,
            "name": item["name"],
            "score": float(D[0][rank]),
            "description": item["description"]
        })
    return results

# -----------------------------
# Main CLI entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Query string for semantic search")
    parser.add_argument("--repo", default="postman-twitter-api", help="Path to repo")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print("Loading documentation...")
    chunks = load_postman_docs(args.repo)
    print(f"Loaded {len(chunks)} chunks.")

    if len(chunks) == 0:
        print("No chunks loaded. Exiting.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding documentation...")
    embeddings = embed_chunks(chunks, model)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("\nSearching...\n")
    results = query_docs(args.query, model, index, chunks, top_k=args.top_k)

    print(json.dumps({"query": args.query, "results": results}, indent=4))

if __name__ == "__main__":
    main()

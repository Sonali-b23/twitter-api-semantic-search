# ML Tasks: Semantic Search & News Narrative Builder

Two small NLP systems in one repo:

1. **Semantic search over Twitter/X API documentation** — natural-language
   query → most relevant API endpoints, using Sentence-BERT embeddings and a
   FAISS index.
2. **News Narrative Builder** — given a topic, retrieves relevant articles
   from a news dataset, builds a chronological timeline, a real generated
   narrative summary, topic clusters, and a relationship graph between
   articles (related / builds-on / contradicts).

## 📦 Installation

```bash
git clone https://github.com/sonali-b23/twitter-api-semantic-search.git
cd twitter-api-semantic-search
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
twitter-api-semantic-search/
├── semantic_search.py
├── narrative_builder.py
├── postman-twitter-api/
│   └── twitter_api_v2.postman_collection.json   # sample Postman collection, ships with the repo
├── news_dataset_sample.json                     # 300-article sample, ships with the repo
├── requirements.txt
└── README.md
```

## Task 1: Semantic Search on Twitter API Documentation

### How it works

1. Recursively walks `postman-twitter-api/` and extracts every endpoint's
   name + description from any Postman collection JSON files found there.
2. Embeds each description with `all-MiniLM-L6-v2` (Sentence-BERT).
3. Indexes the embeddings in a FAISS flat index.
4. Embeds the query the same way and retrieves the top-k nearest endpoints
   by similarity.

### Usage

```bash
python semantic_search.py --query "How do I search recent tweets?"
python semantic_search.py --query "How do I stream tweets in real time?" --top_k 3
python semantic_search.py --query "How do I look up a user by username?" --output results.json
```

### About the bundled Postman collection

`postman-twitter-api/` ships with a small, self-authored sample collection
(17 endpoints across Tweet lookup, Search, Filtered stream, Users, Spaces,
Lists, and Compliance) so the search actually works right after cloning,
with no extra setup. **This folder used to be committed as a broken git
submodule reference with no data behind it — running the search always
returned zero results.** That's fixed now; the folder holds real files.

To search over a fuller, live-updated Twitter API collection instead,
export one from [Postman's public API](https://learning.postman.com/docs/collections/using-postman-for-api-testing/)
as a `.postman_collection.json` file and drop it into this folder — the
loader walks the whole directory recursively, so multiple collection files
work fine together.

## Task 2: News Narrative Builder

### How it works

1. **Filter** — load the dataset, keep articles with `source_rating > 8`.
2. **Retrieve** — embed all articles and the query topic, keep the top-k
   most similar articles.
3. **Timeline** — sort the retrieved articles chronologically.
4. **Narrative summary** — a real generated paragraph (not a headline
   concatenation): the chronological headlines are fed to `google/flan-t5-small`
   with a summarization prompt. If `transformers`/`torch` aren't installed
   or the model can't be downloaded, this falls back to a plain headline
   join with a warning printed, so the script never hard-fails.
5. **Clusters** — group the retrieved articles into `n_clusters` topic
   groups via agglomerative clustering on their embeddings.
6. **Relationship graph** — for every pair of articles above a similarity
   threshold (0.7), the graph gets a real, distinguishing edge label:
   - `contradicts` — a small NLI cross-encoder
     (`cross-encoder/nli-deberta-v3-small`) classifies the pair; used only
     as a fallback labeling scheme, not a hardcoded string.
   - `builds_on` — the pair is similar *and* published within 90 days of
     each other, with `earlier`/`later` recorded on the edge.
   - `related` — similar, but not classified as either of the above (e.g.
     similar topic, published far apart in time).

   If the NLI model can't be loaded, edges fall back to `related` /
   `builds_on` only, with a warning — the graph is still produced, just
   with one fewer distinction.

### Usage

```bash
python narrative_builder.py --topic "Hyderabad Metro"
python narrative_builder.py --topic "cricket" --top_k 30 --n_clusters 4 --output cricket_narrative.json
```

By default this runs against the bundled `news_dataset_sample.json` (300
real articles, sampled from a larger 36k-article dataset). To use the full
dataset, pass `--dataset path/to/your_full_dataset.json` — the full file
isn't committed here because it's ~81MB; see "Notes on the dataset" below.

### Output format

```json
{
  "topic": "...",
  "narrative_summary": "...",
  "timeline": [...],
  "clusters": [...],
  "graph": {"nodes": [...], "links": [...]}
}
```

## 📌 Notes on the dataset

The original 81MB / 36,483-article dataset isn't committed to this repo —
committing a file that large bloats every clone. A 300-article sample
(filtered to `source_rating > 8`, deduplicated by title) ships instead so
the narrative builder works out of the box. If you have your own news
dataset in the same shape (`title`, `story`, `url`, `published_at`,
`source_rating`), point `--dataset` at it directly.

## ⚠️ Known limitations

- `builds_on` / `contradicts` labeling is heuristic (time-window +
  cross-encoder classification), not ground-truth fact-checking — treat it
  as a reasonable signal, not a verified claim.
- The narrative summary quality depends on `flan-t5-small`, a small model
  chosen for CPU-friendliness; swapping in a larger model (e.g.
  `flan-t5-base` or an API-based LLM) in `_get_summarizer()` will improve
  fluency at the cost of speed/compute.
- `semantic_search.py`'s bundled Postman collection is a representative
  sample, not the complete official Twitter/X API surface.

## 🛠️ Tech Stack

- Python
- Sentence-BERT (`sentence-transformers`)
- FAISS
- `transformers` / `flan-t5-small` (narrative generation)
- Cross-encoder NLI (`cross-encoder/nli-deberta-v3-small`) (contradiction detection)
- scikit-learn, NetworkX, NumPy

## 📊 Future Improvements

- Add evaluation metrics (precision@k for search).
- Try a larger summarization model for narrative generation quality.
- Visualize the narrative graph interactively (e.g. with `pyvis` or a small
  Streamlit app).
- Deploy semantic search as a small web API/UI.

## Conclusion

Two practical applications of embeddings-based NLP: information retrieval
(semantic search over API docs) and story understanding (narrative
generation, clustering, and relationship extraction over a news corpus).

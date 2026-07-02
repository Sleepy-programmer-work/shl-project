import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_DICT = {}
CATALOG_LIST = []
CATALOG_TEXTS = []
VECTORIZER = None
TFIDF_MATRIX = None

def load_and_clean_data():
    global CATALOG_DICT, CATALOG_LIST, CATALOG_TEXTS, VECTORIZER, TFIDF_MATRIX

    if not os.path.exists("catalog.json"):
        print("Warning: catalog.json file not found.")
        return

    with open("catalog.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_catalog = []

    # I discovered that catalog.json has trailing spaces like "entity_id " and extra whitespace in values that broke my matching, so I have to strip everything on startup.
    for item in raw_data:
        cleaned_item = {}
        for key, val in item.items():
            clean_key = key.strip()

            if isinstance(val, str):
                clean_val = val.strip()
            elif isinstance(val, list):
                clean_val = []
                for element in val:
                    if isinstance(element, str):
                        clean_val.append(element.strip())
                    else:
                        clean_val.append(element)
            else:
                clean_val = val

            cleaned_item[clean_key] = clean_val

        cleaned_catalog.append(cleaned_item)

    CATALOG_DICT = {}
    CATALOG_LIST = []
    CATALOG_TEXTS = []

    for item in cleaned_catalog:
        entity_id = item.get("entity_id")
        if not entity_id:
            continue

        CATALOG_DICT[str(entity_id)] = item
        CATALOG_LIST.append(item)

        name = item.get("name", "")
        description = item.get("description", "")
        keys = item.get("keys", [])

        keys_str = ", ".join(keys)
        text_blob = f"{name}. {description}. Keys: {keys_str}"
        CATALOG_TEXTS.append(text_blob)

    # I choose TF-IDF because it's fast and efficient for this task.
    if CATALOG_TEXTS:
        VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        TFIDF_MATRIX = VECTORIZER.fit_transform(CATALOG_TEXTS)
        print(f"TF-IDF index built: {len(CATALOG_LIST)} assessments indexed.")
    else:
        print("Warning: no catalog items to index.")

def search(query: str, top_k: int = 25):
    global CATALOG_LIST, VECTORIZER, TFIDF_MATRIX

    if VECTORIZER is None or TFIDF_MATRIX is None:
        return []

    query_vec = VECTORIZER.transform([query])

    scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = CATALOG_LIST[idx]
        entity_id = item.get("entity_id", "")
        name = item.get("name", "")
        description = item.get("description", "")
        keys = item.get("keys", [])

        results.append({
            "entity_id": entity_id,
            "name": name,
            "description": description,
            "keys": keys
        })

    return results

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# global variables to cache everything in memory on startup
CATALOG_DICT = {}
CATALOG_LIST = []
CATALOG_TEXTS = []
VECTORIZER = None
TFIDF_MATRIX = None

# load the catalog, clean dirty keys/values, and build the tfidf index
def load_and_clean_data():
    global CATALOG_DICT, CATALOG_LIST, CATALOG_TEXTS, VECTORIZER, TFIDF_MATRIX

    # check the catalog file is present before trying to open it
    if not os.path.exists("catalog.json"):
        print("Warning: catalog.json file not found.")
        return

    # read the raw json file from disk
    with open("catalog.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_catalog = []

    # loop through each item and strip spaces from keys and string values
    # the dataset has dirty trailing spaces like "entity_id " that break lookups
    for item in raw_data:
        cleaned_item = {}
        for key, val in item.items():
            # strip spaces from the key itself
            clean_key = key.strip()

            # strip spaces from plain string values
            if isinstance(val, str):
                clean_val = val.strip()
            # strip spaces from every string element inside list values
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

    # reset our global caches before filling them
    CATALOG_DICT = {}
    CATALOG_LIST = []
    CATALOG_TEXTS = []

    # build the catalog dict and the list of text strings to embed
    for item in cleaned_catalog:
        entity_id = item.get("entity_id")
        if not entity_id:
            continue

        # store item keyed by entity_id string for fast lookups later
        CATALOG_DICT[str(entity_id)] = item
        CATALOG_LIST.append(item)

        # combine name, description, and keys into one text blob for tfidf indexing
        name = item.get("name", "")
        description = item.get("description", "")
        keys = item.get("keys", [])

        keys_str = ", ".join(keys)
        text_blob = f"{name}. {description}. Keys: {keys_str}"
        CATALOG_TEXTS.append(text_blob)

    # fit the tfidf vectorizer on all catalog texts and build the sparse matrix
    # tfidf uses almost no memory compared to sentence-transformers (~5MB vs ~400MB)
    if CATALOG_TEXTS:
        VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        TFIDF_MATRIX = VECTORIZER.fit_transform(CATALOG_TEXTS)
        print(f"TF-IDF index built: {len(CATALOG_LIST)} assessments indexed.")
    else:
        print("Warning: no catalog items to index.")

# search the catalog using tfidf cosine similarity against the query
def search(query: str, top_k: int = 25):
    global CATALOG_LIST, VECTORIZER, TFIDF_MATRIX

    # return empty list if catalog hasn't been loaded yet
    if VECTORIZER is None or TFIDF_MATRIX is None:
        return []

    # transform the query into the same tfidf vector space
    query_vec = VECTORIZER.transform([query])

    # calculate cosine similarity between the query and all catalog texts
    scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()

    # sort by similarity descending and take the top_k indexes
    top_indices = scores.argsort()[::-1][:top_k]

    # build the results list from the top matching catalog items
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

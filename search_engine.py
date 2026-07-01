import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# global variables to cache data in memory so we don't reload on every request
CATALOG_DICT = {}
CATALOG_LIST = []
CATALOG_EMBEDDINGS = None
MODEL = None

# load the catalog json file, strip dirty spaces from keys and values, and embed the texts
def load_and_clean_data():
    global CATALOG_DICT, CATALOG_EMBEDDINGS, CATALOG_LIST, MODEL

    # load the sentence transformer model for embedding
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")

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
    texts_to_embed = []

    # build the catalog dict and the list of text strings to embed
    for item in cleaned_catalog:
        entity_id = item.get("entity_id")
        if not entity_id:
            continue

        # store item keyed by entity_id string for fast lookups later
        CATALOG_DICT[str(entity_id)] = item
        CATALOG_LIST.append(item)

        # combine name, description, and keys into one text blob for embedding
        name = item.get("name", "")
        description = item.get("description", "")
        keys = item.get("keys", [])

        keys_str = ", ".join(keys)
        text_blob = f"{name}. {description}. Keys: {keys_str}"
        texts_to_embed.append(text_blob)

    # encode all the text blobs into a numpy matrix of embeddings
    if texts_to_embed:
        CATALOG_EMBEDDINGS = MODEL.encode(texts_to_embed, convert_to_numpy=True)
    else:
        CATALOG_EMBEDDINGS = np.array([])

# search the catalog using cosine similarity between query and product embeddings
def search(query: str, top_k: int = 15):
    global CATALOG_DICT, CATALOG_EMBEDDINGS, CATALOG_LIST, MODEL

    # return empty list if catalog hasn't been loaded yet
    if CATALOG_EMBEDDINGS is None or len(CATALOG_EMBEDDINGS) == 0:
        return []

    # embed the user query into a vector
    query_embedding = MODEL.encode([query], convert_to_numpy=True)[0]

    # calculate the L2 norm of all product embeddings and the query embedding
    norms_prod = np.linalg.norm(CATALOG_EMBEDDINGS, axis=1)
    norm_query = np.linalg.norm(query_embedding)

    # replace zero norms with a tiny number to avoid division by zero
    norms_prod[norms_prod == 0] = 1e-10
    if norm_query == 0:
        norm_query = 1e-10

    # calculate cosine similarity using dot product divided by the norms
    similarities = np.dot(CATALOG_EMBEDDINGS, query_embedding) / (norms_prod * norm_query)

    # sort by similarity descending and take the top_k indexes
    top_indices = np.argsort(similarities)[::-1][:top_k]

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

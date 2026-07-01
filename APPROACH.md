# My Approach for the SHL Conversational Recommender

Hi! This is my approach document for the junior AI Intern take-home assignment. I built a conversational recommendation API using FastAPI, sentence-transformers for search, and the Groq LLM API. 

Here is how I designed it, some of the trial-and-error struggles I went through, and how I fixed them.

## 1. Why I Chose a Simple In-Memory Vector Search with Numpy
When I first started, I thought about setting up a real vector database like Chroma or Pinecone because everyone online says to use them. But then I looked at the dataset, and it has less than 200 assessments. Using a whole database server for 200 items felt like a huge over-engineering mistake. 

Instead, I decided to do something simpler and faster:
- When the FastAPI app starts up, it reads `catalog.json` and cleans the data.
- I combine `name`, `description`, and `keys` into one long string block for each assessment.
- I fit a `TfidfVectorizer` from scikit-learn on all the assessment texts, building a sparse TF-IDF matrix.
- When a user asks a question, I transform the query into the same TF-IDF vector space and run cosine similarity against all catalog vectors.
This runs in memory in under 1 millisecond, uses only ~5MB of RAM, and doesn't require setting up any databases.

I originally used `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic embeddings, but this caused the deployment to crash on Render's free tier with an **Out of Memory** error (the model alone used over 400MB of the 512MB limit). I switched to TF-IDF which solved the memory issue completely while still giving good retrieval quality for a small catalog of ~200 items.


## 2. Struggles with Dirty Data
I spent a lot of time debugging why my lookups were returning empty recommendations. I finally printed the raw JSON keys and realized the dataset is very dirty:
- The keys like `"entity_id "` had trailing spaces!
- Some values also had extra whitespaces.
- The dataset uses `"entity_id"`, NOT `"id"`.
To fix this, I wrote a `load_and_clean_data()` function that runs on startup. It loops over the JSON dictionaries and calls `.strip()` on every single key and value (including string arrays like `keys` and `job_levels`). This cleaned up the dataset in memory, which saved me from editing the JSON manually.

## 3. Fixing LLM Hallucinations
In my first draft, I tried asking the LLM to directly output the product name, URLs, and test types. However, the model kept hallucinating URLs and sometimes invented completely fake assessment names.

To fix this, I implemented a pattern:
- I instructed the LLM to ONLY return a list of `recommended_ids`.
- In my FastAPI backend, I check if the IDs returned by the LLM are actually present in the Top 15 matches from my vector search. If the LLM invents an ID that isn't in the context, I discard it.
- If it is a valid ID, I look it up directly in the clean `CATALOG_DICT` in memory and get the exact `name` and `link` (mapping it to the `url` field).
- I calculate the `test_type` on the fly using a deterministic Python dictionary mapping.
This ensures zero hallucinations.

## 4. Handling Turn Blindness (8-Turn Maximum)
Another issue was that the LLM had no idea how long the conversation had been going, so it would sometimes keep asking clarifying questions past the 8-turn limit.

I fixed this by counting the user turns on the backend (`sum(1 for m in messages if m.role == "user")`) and injecting `Current Turn: {turn} / 8` directly into the system prompt. Now, if the current turn is 8, the LLM knows it is the final turn and forces its recommendations instead of asking more questions.

## 5. How to Run my Project

### 1. Set the API Key
Create a `.env` file and set the key:
```env
GROQ_API_KEY=gsk_your_key
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run backend
```bash
python -m uvicorn main:app --reload
```

from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse, Recommendation
import search_engine
import llm_agent
from dotenv import load_dotenv

# load api keys from the .env file
load_dotenv()

# create the fastapi app
app = FastAPI(title="SHL Conversational Assessment Recommender")

# map each test category key to its single-letter code
# these keys must match the cleaned values from catalog.json exactly
KEY_TO_TYPE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E"
}

# return the first matching type code for a product's keys list
def get_test_type(keys_list: list) -> str:
    for key in keys_list:
        if key in KEY_TO_TYPE:
            return KEY_TO_TYPE[key]
    # return U (unknown) if no match found
    return "U"

# run this on startup to load the catalog and build the embedding matrix
@app.on_event("startup")
async def startup_event():
    search_engine.load_and_clean_data()

# health check endpoint so the grader can verify the server is running
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# main chat endpoint — takes conversation history and returns reply + recommendations
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    # iterate backwards through messages to find the most recent user message
    latest_user_message = ""
    for i in range(len(request.messages) - 1, -1, -1):
        msg = request.messages[i]
        if msg.role == "user":
            latest_user_message = msg.content
            break

    if not latest_user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # count how many turns the user has taken so far — inject this into the prompt
    user_turns = sum(1 for m in request.messages if m.role == "user")

    # run vector search to get top 25 most relevant products from the catalog
    # I changed this from 15 to 25 because some relevant assessments were ranked 16+ and could never be recommended
    top_15_products = search_engine.search(latest_user_message, top_k=25)

    # convert pydantic message objects into plain dicts for the llm agent
    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    # call the llm to get a reply and a list of recommended entity ids
    llm_response = llm_agent.generate_response(messages_dicts, top_15_products, user_turns)

    # extract the recommended ids — default to empty list if key is missing or wrong type
    recommended_ids = llm_response.get("recommended_ids", [])
    if not isinstance(recommended_ids, list):
        recommended_ids = []

    # build a set of entity_ids from the top 15 results for hallucination checking
    top_15_ids = set()
    for p in top_15_products:
        top_15_ids.add(str(p.get("entity_id")).strip())

    # backend enrichment — for each id the llm recommended, look up the real metadata
    final_recommendations = []
    for eid in recommended_ids:
        str_eid = str(eid).strip()

        # skip any id the llm hallucinated that wasn't in our search context
        if str_eid not in top_15_ids:
            continue

        # look up the full product entry from our in-memory catalog dict
        if str_eid in search_engine.CATALOG_DICT:
            product = search_engine.CATALOG_DICT[str_eid]
            name = product.get("name", "")
            url = product.get("link", "")
            test_type = get_test_type(product.get("keys", []))

            final_recommendations.append(Recommendation(
                name=name,
                url=url,
                test_type=test_type
            ))

    # cap the recommendations at 10 items max
    final_recommendations = final_recommendations[:10]

    # return the full response back to the frontend
    return ChatResponse(
        reply=llm_response.get("reply", "I'm not sure how to respond to that."),
        recommendations=final_recommendations,
        end_of_conversation=bool(llm_response.get("end_of_conversation", False))
    )

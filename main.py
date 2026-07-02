from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse, Recommendation
import search_engine
import llm_agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SHL Conversational Assessment Recommender")

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

def get_test_type(keys_list: list) -> str:
    for key in keys_list:
        if key in KEY_TO_TYPE:
            return KEY_TO_TYPE[key]
    return "U"

@app.on_event("startup")
async def startup_event():
    search_engine.load_and_clean_data()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    latest_user_message = ""
    for i in range(len(request.messages) - 1, -1, -1):
        msg = request.messages[i]
        if msg.role == "user":
            latest_user_message = msg.content
            break

    if not latest_user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    user_turns = sum(1 for m in request.messages if m.role == "user")

    # I changed this from 15 to 25 because some relevant assessments were ranked 16+ and could never be recommended
    top_25_products = search_engine.search(latest_user_message, top_k=25)

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    llm_response = llm_agent.generate_response(messages_dicts, top_25_products, user_turns)

    recommended_ids = llm_response.get("recommended_ids", [])
    if not isinstance(recommended_ids, list):
        recommended_ids = []

    top_25_ids = set()
    for p in top_25_products:
        top_25_ids.add(str(p.get("entity_id")).strip())

    final_recommendations = []
    for eid in recommended_ids:
        str_eid = str(eid).strip()

        if str_eid not in top_25_ids:
            continue

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

    final_recommendations = final_recommendations[:10]

    return ChatResponse(
        reply=llm_response.get("reply", "I'm not sure how to respond to that."),
        recommendations=final_recommendations,
        end_of_conversation=bool(llm_response.get("end_of_conversation", False))
    )

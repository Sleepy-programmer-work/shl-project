import os
import json
from groq import Groq
from typing import List, Dict

# I chose llama-3.1-8b-instant because it's fast enough to comfortably stay under the 30s timeout limit.
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT_TEMPLATE = """You are an SHL Assessment Recommender Agent. Your goal is to help hiring managers find the right SHL assessments from the provided catalog.
Current Turn: {turn} / 8

RULES:
1. You ONLY discuss SHL assessments. Refuse general hiring advice, legal questions, or off-topic chat politely.
2. If the user's intent is vague, ask 1 clarifying question. Do NOT recommend yet.
3. If you have enough context, recommend between 1 and 10 assessments.
4. NEVER invent assessments. ONLY use the `entity_id` from the <available_catalog>.
5. If the Current Turn is 7 or 8, you MUST force your final recommendation and set "end_of_conversation" to true.
6. If comparing tests, use ONLY catalog descriptions.
7. If refining, update the shortlist.

OUTPUT FORMAT (Strict JSON):
{{
  "thought_process": "Brief reasoning...",
  "reply": "Your conversational response...",
  "recommended_ids": ["entity_id_1", "entity_id_2"],
  "end_of_conversation": false
}}

<available_catalog>
{context}
</available_catalog>"""

def generate_response(messages: List[Dict[str, str]], top_products: List[Dict], turn: int):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {
            "thought_process": "Missing API Key",
            "reply": "I apologize, but the AI service is currently unavailable.",
            "recommended_ids": [],
            "end_of_conversation": False
        }

    client = Groq(api_key=api_key)

    context_lines = []
    for p in top_products:
        keys_str = ", ".join(p.get("keys", []))
        line = f"- ID: {p.get('entity_id')} | Name: {p.get('name')} | Desc: {p.get('description')} | Keys: {keys_str}"
        context_lines.append(line)

    context_string = "\n".join(context_lines)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(turn=turn, context=context_string)

    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        api_messages.append({"role": msg.get("role"), "content": msg.get("content")})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=api_messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )

        response_json = json.loads(response.choices[0].message.content)
        return response_json
    except Exception as e:
        print(f"Groq API Error: {str(e)}")
        return {
            "thought_process": f"Error: {str(e)}",
            "reply": "I encountered an internal error. Could you repeat that?",
            "recommended_ids": [],
            "end_of_conversation": False
        }
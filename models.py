from pydantic import BaseModel
from typing import List

# Message model representing a single turn in the conversation
class Message(BaseModel):
    role: str      # 'user', 'assistant', or 'system'
    content: str   # the text of the message

# ChatRequest model representing the payload sent from the frontend
class ChatRequest(BaseModel):
    messages: List[Message]  # array of messages in the chat history

# Recommendation model representing a recommended test
class Recommendation(BaseModel):
    name: str       # name of the test
    url: str        # link to the test details
    test_type: str  # mapped test type (K, P, S, etc.)

# ChatResponse model representing the API response
class ChatResponse(BaseModel):
    reply: str                            # conversational response
    recommendations: List[Recommendation] # list of matching recommendations
    end_of_conversation: bool             # flag indicating whether conversation is complete

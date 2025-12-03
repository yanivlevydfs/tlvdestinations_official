# models/chat_query.py
from pydantic import BaseModel

class ChatQuery(BaseModel):
    question: str

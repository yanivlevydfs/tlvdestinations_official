# rag_agent.py


import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any

from google import genai
from google.adk.agents import Agent
from google.adk.tools import function_tool

# Gemini models
EMBED_MODEL = "models/embedding-001"
GEN_MODEL = "models/gemini-1.5-flash"

# Set environment variables
genai.configure(api_key="AIzaSyBxOJwavtKVB9gJvA2OoAsKw90GogBNdZs")

# Globals
INDEX = None
DOCS = []
VECTORS = None


def build_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    docs = []
    for idx, row in df.iterrows():
        text = (
            f"IATA: {row['IATA']}\n"
            f"Name: {row['Name']}\n"
            f"City: {row['City']}\n"
            f"Country: {row['Country']}\n"
            f"Airlines: {', '.join(row['Airlines']) if isinstance(row['Airlines'], list) else row['Airlines']}\n"
            f"Distance: {row['Distance_km']} km\n"
            f"Flight Time: {row['FlightTime_hr']} hours"
        )
        docs.append({
            "id": str(idx),
            "content": text,
            "meta": {
                "IATA": row["IATA"],
                "City": row["City"],
                "Country": row["Country"]
            }
        })
    return docs


def build_index(docs: List[Dict[str, Any]]):
    global INDEX, VECTORS
    contents = [doc["content"] for doc in docs]

    embeddings = genai.embed_content(
        model=EMBED_MODEL,
        content=contents,
        task_type="retrieval_document"
    ).embeddings

    vectors = np.array(embeddings, dtype='float32')
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    INDEX = index
    VECTORS = vectors


@function_tool(name="get_contexts", description="Retrieves the most relevant airport records.")
def get_contexts(query: str, top_k: int = 5) -> Dict[str, Any]:
    q_embed = genai.embed_content(
        model=EMBED_MODEL,
        content=[query],
        task_type="retrieval_query"
    ).embeddings[0]

    q_vector = np.array([q_embed], dtype='float32')
    scores, indices = INDEX.search(q_vector, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(DOCS):
            results.append(DOCS[idx]["content"])
    return {
        "contexts": results
    }


def init_agent(df: pd.DataFrame) -> Agent:
    global DOCS
    DOCS = build_documents(df)
    build_index(DOCS)

    return Agent(
        name="flights_rag_agent",
        model=GEN_MODEL,
        instruction=(
            "You are a helpful assistant who answers questions about international flights from TLV airport. "
            "Use the provided airport and airline data (from the 'get_contexts' tool) to answer questions. "
            "Do not answer from your own knowledge — only use the provided data. If no relevant data, say so."
        ),
        tools=[get_contexts],
    )


def chat(agent: Agent):
    print("💬 Gemini Chat (type 'exit' to quit)")
    while True:
        q = input("👤: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        result = agent.run(q)
        print(f"🤖: {result.text}\n")

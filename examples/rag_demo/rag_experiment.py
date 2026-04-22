# /// script
# dependencies = [
#   "qdrant-client",
#   "httpx",
#   "apple-fm-cli",
# ]
# ///
import sys
import os
import asyncio
import json
import apple_fm_sdk as fm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def run_rag_demo():
    # 1. Initialize Qdrant and Apple FM
    client = QdrantClient("localhost", port=6333)
    collection_name = "apple_fm_knowledge"
    
    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    
    # 2. Prepare "Private" Knowledge
    documents = [
        {
            "id": 1,
            "text": "The Project Titan manual states that the emergency override code is 'APPLE-BLUE-2026'.",
            "metadata": {"source": "manual_v1"}
        },
        {
            "id": 2,
            "text": "The secure server room is located on the 4th floor, behind the glass partition in Sector 7G.",
            "metadata": {"source": "building_specs"}
        },
        {
            "id": 3,
            "text": "Employee of the month for April 2026 is Sarah Jenkins from the Neural Engine optimization team.",
            "metadata": {"source": "internal_memo"}
        }
    ]
    
    print("Indexing documents into Qdrant...")
    points = []
    for doc in documents:
        # Use our new native embedding!
        vector = fm.get_sentence_embedding(doc["text"])
        points.append(PointStruct(
            id=doc["id"],
            vector=vector,
            payload={"text": doc["text"], "metadata": doc["metadata"]}
        ))
    
    client.upsert(collection_name=collection_name, points=points)
    print("Indexing complete.\n")
    
    # 3. RAG Query
    query = "Where is the secure server room located and what is the emergency code?"
    print(f"Query: {query}")
    
    # Get query embedding
    query_vector = fm.get_sentence_embedding(query)
    
    # Search in Qdrant
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=2
    ).points
    
    # 4. Augment and Generate
    context = "\n".join([res.payload["text"] for res in search_result])
    print("\n--- Retrieved Context ---")
    print(context)
    print("-------------------------\n")
    
    prompt = f"""Use the following pieces of retrieved context to answer the question. 
If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {query}
Answer:"""

    # Use Apple Intelligence to generate the final answer
    print("Generating answer with Apple Intelligence...")
    session = fm.LanguageModelSession()
    
    # Using a simple async wrapper for the demo
    import asyncio
    async def get_response():
        return await session.respond(prompt)
    
    answer = asyncio.run(get_response())
    print(f"\nFinal Answer:\n{answer}")

if __name__ == "__main__":
    run_rag_demo()

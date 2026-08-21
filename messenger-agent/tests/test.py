# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import hmac
import hashlib
import json
from typing import Optional
from langgraph.graph import StateGraph
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# Store your bot's secret (from Chatwoot agent bot settings)
BOT_SECRET = os.getenv("CHATWOOT_BOT_SECRET", "your-secret-key")

# Your Chatwoot API credentials
CHATWOOT_API_KEY = os.getenv("CHATWOOT_API_KEY")
CHATWOOT_BASE_URL = os.getenv("CHATWOOT_BASE_URL", "https://your-chatwoot-domain.com")

# ============= Define Your LangGraph Agent =============
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize your LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

def process_message(state: State):
    """Process message with LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("agent", process_message)
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)

agent = graph_builder.compile()

# ============= Webhook Handler =============
class ChatwootMessage(BaseModel):
    event: str
    id: int
    content: str
    conversation_id: int
    account_id: int
    inbox_id: int
    sender: Optional[dict] = None

def verify_webhook_signature(payload: str, signature: str) -> bool:
    """Verify Chatwoot webhook signature"""
    expected_signature = hmac.new(
        BOT_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected_signature)

@app.post("/webhook")
async def handle_chatwoot_webhook(request: Request):
    """
    Webhook endpoint that receives messages from Chatwoot
    """
    # Get the raw body for signature verification
    body = await request.body()
    signature = request.headers.get("X-Chatwoot-Webhook-Signature", "")
    
    # Verify signature
    if not verify_webhook_signature(body.decode(), signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        data = json.loads(body)
        message_data = ChatwootMessage(**data)
        
        # Only process incoming messages (not outgoing)
        if message_data.event != "message_created":
            return JSONResponse({"status": "ignored"})
        
        # Get the message content
        user_message = message_data.content
        
        print(f"Received message: {user_message}")
        print(f"Conversation ID: {message_data.conversation_id}")
        
        # ============= Process with LangGraph =============
        # Initialize state with the user message
        state = {"messages": [{"role": "user", "content": user_message}]}
        
        # Run your LangGraph agent
        result = agent.invoke(state)
        
        # Extract the AI response
        ai_response = result["messages"][-1].content
        
        print(f"Agent response: {ai_response}")
        
        # ============= Send Response Back to Chatwoot =============
        await send_message_to_chatwoot(
            account_id=message_data.account_id,
            conversation_id=message_data.conversation_id,
            content=ai_response
        )
        
        return JSONResponse({"status": "success", "response": ai_response})
        
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def send_message_to_chatwoot(
    account_id: int,
    conversation_id: int,
    content: str
):
    """
    Send the bot's response back to Chatwoot as an outgoing message
    """
    url = f"{CHATWOOT_BASE_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    
    headers = {
        "Authorization": f"Bearer {CHATWOOT_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "content": content,
        "message_type": "outgoing",  # Bot message
        "private": False
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
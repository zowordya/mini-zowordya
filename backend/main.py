from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

app = FastAPI(title="Gemini Web Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class GeminiRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    max_output_tokens: int = 8192

@app.get("/")
async def root():
    return {"message": "Gemini Web Assistant API is running"}

@app.post("/api/generate")
async def generate_content(request: GeminiRequest):
    """
    Proxy endpoint for Gemini API generate content
    """
    try:
        # Prepare the request to Gemini API
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        # Construct the request payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": request.prompt
                }]
            }],
            "generationConfig": {
                "temperature": request.temperature,
                "topK": request.top_k,
                "topP": request.top_p,
                "maxOutputTokens": request.max_output_tokens,
            }
        }
        
        # Make the request to Gemini API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{gemini_url}?key={GEMINI_API_KEY}",
                json=payload,
                timeout=30.0
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                raise HTTPException(status_code=response.status_code, detail=error_message)
            
            # Return the Gemini API response
            return response.json()
            
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# For production, serve the static frontend files
@app.get("/{full_path:path}")
async def serve_frontend(request: Request):
    return {"message": "Frontend would be served here in production"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

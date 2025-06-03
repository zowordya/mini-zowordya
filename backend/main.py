from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os
import logging
import traceback
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-api")

# Load environment variables
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable not set")
else:
    # Log partial key for verification (safely)
    masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 8 else "***"
    logger.info(f"API Key loaded: {masked_key}")

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
    logger.info("Root endpoint accessed")
    return {"message": "Gemini Web Assistant API is running"}

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    logger.info("Health check endpoint accessed")
    return {"status": "ok", "api_key_configured": bool(GEMINI_API_KEY), "model": "gemini-2.0-flash"}

@app.post("/api/generate")
async def generate_content(request: GeminiRequest):
    """
    Proxy endpoint for Gemini API generate content
    """
    logger.info(f"Generate content request received: prompt length {len(request.prompt)}")
    
    try:
        # Updated to use gemini-2.0-flash model
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
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
        
        logger.info(f"Sending request to Gemini API: {gemini_url} (using gemini-2.0-flash model)")
        
        # Make the request to Gemini API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{gemini_url}?key={GEMINI_API_KEY}",
                    json=payload,
                    timeout=30.0
                )
                
                logger.info(f"Gemini API response status: {response.status_code}")
                
                # Check if the request was successful
                if response.status_code != 200:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", "Unknown error")
                    logger.error(f"Gemini API error: {error_message}")
                    logger.error(f"Full error response: {json.dumps(error_data)}")
                    raise HTTPException(status_code=response.status_code, detail=error_message)
                
                # Return the Gemini API response
                result = response.json()
                logger.info("Successfully received and returning Gemini API response")
                return result
                
            except httpx.RequestError as e:
                logger.error(f"Request error to Gemini API: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

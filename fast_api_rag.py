from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import os
import shutil
import logging
import time
import uvicorn
from dotenv import load_dotenv
import asyncio
import json

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LightRAG API", version="1.2")

# ------------------------ Configuration ------------------------

# Ensure the environment variable for OpenAI is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables.")
    raise EnvironmentError("OPENAI_API_KEY is required.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------------ Initialize LightRAG ------------------------

class InitializeLightRAG:
    def __init__(self, zip_path: str = "book_backup.zip", extraction_path: str = "book_data/"):
        self.zip_path = zip_path
        self.extraction_path = extraction_path
        self.rag = None
        self.initialize()

    def initialize(self):
        if not os.path.exists(self.extraction_path):
            if not os.path.exists(self.zip_path):
                logger.error(f"Zip file {self.zip_path} does not exist.")
                raise FileNotFoundError(f"{self.zip_path} not found.")
            shutil.unpack_archive(self.zip_path, self.extraction_path)
            logger.info(f"📦 Book folder unzipped to: {self.extraction_path}")
        self.rag = LightRAG(
            working_dir=self.extraction_path,
            llm_model_func=gpt_4o_mini_complete
        )
        logger.info("🔄 LightRAG system initialized.")
    
    async def aquery(self, query: str, param: QueryParam):
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self.rag.query, query, param)
        return response

# Initialize LightRAG once when the API starts
initialize_lightrag = InitializeLightRAG()

# ------------------------ Request and Response Models ------------------------

class GenerateResponseRequest(BaseModel):
    prompt: str = Field(..., example="What are the benefits of renewable energy?")
    number_of_responses: int = Field(..., ge=1, le=5, example=2)
    response_types: List[str] = Field(..., example=["positive", "negative"])
    search_mode: str = Field(..., example="hybrid", description="Options: naive, local, global, hybrid")

class GeneratedResponse(BaseModel):
    response_type: str
    response_text: str
    latency_seconds: float

class GenerateResponseResponse(BaseModel):
    responses: List[GeneratedResponse]
    total_latency_seconds: float

# ------------------------ API Endpoint ------------------------

@app.post("/generate_response_informed", response_model=GenerateResponseResponse)
async def generate_response_informed(request: GenerateResponseRequest):
    if not request.prompt.strip():
        logger.warning("Empty prompt received.")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if len(request.response_types) != request.number_of_responses:
        logger.warning("Number of response types does not match number of responses requested.")
        raise HTTPException(
            status_code=400,
            detail="The number of response types must match the number of responses requested."
        )

    logger.info(f"Received request with prompt: {request.prompt[:50]}..., "
                f"number_of_responses: {request.number_of_responses}, "
                f"response_types: {request.response_types}, "
                f"search_mode: {request.search_mode}")

    # Define the system prompt with JSON instruction
    system_prompt = (
        "As Todd, respond to the following question in a conversational manner, "
        "keeping each response under 15 words for brevity and relevance. "
        "Focus on providing honest and personal answers that align with my perspective in the story. "
        "Provide the responses in JSON format as a list of objects, each with 'response_type' and 'response_text' fields. "
        "Return only the JSON without any additional text."
    )

    start_time = time.time()
    try:
        # Construct the system query by combining system_prompt with the user prompt
        system_query = (
            f"{system_prompt}\n\n"
            f"Question: {request.prompt}\n\n"
            f"Provide {request.number_of_responses} responses as follows:\n"
        )
        for i, resp_type in enumerate(request.response_types, start=1):
            system_query += f"{i}. {resp_type.capitalize()} response:\n"

        # Query LightRAG with the specified search mode
        response = await initialize_lightrag.aquery(system_query, QueryParam(mode=request.search_mode))
        
        # Debug logging
        logger.debug(f"Type of response: {type(response)}")
        logger.debug(f"Content of response: {response}")

        # Parse the response if it's a string
        if isinstance(response, str):
            try:
                response = json.loads(response)
                logger.debug("Parsed JSON string into dictionary.")
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON string.", exc_info=True)
                raise HTTPException(status_code=500, detail="Invalid response format from LightRAG.")

        # Ensure responses are in the expected format (list of dicts with 'response_type' and 'response_text' keys)
        responses = response.get('responses', [])
        if not isinstance(responses, list):
            logger.error("Responses key is not a list.")
            raise HTTPException(status_code=500, detail="Invalid response structure from LightRAG.")

        # Validate each response item
        for resp in responses:
            if not isinstance(resp, dict):
                logger.error("Each response should be a dictionary.")
                raise HTTPException(status_code=500, detail="Invalid response item format from LightRAG.")
            if 'response_type' not in resp or 'response_text' not in resp:
                logger.error("Missing 'response_type' or 'response_text' in response item.")
                raise HTTPException(status_code=500, detail="Incomplete response item from LightRAG.")

        # Validate and structure responses
        generated_responses = []
        for idx, resp in enumerate(responses):
            generated_responses.append(GeneratedResponse(
                response_type=resp.get('response_type', 'unknown'),
                response_text=resp.get('response_text', ''),
                latency_seconds=round(time.time() - start_time, 2)
            ))

        total_latency = round(time.time() - start_time, 2)
        logger.info(f"Generated {len(generated_responses)} responses in {total_latency} seconds.")

        return GenerateResponseResponse(responses=generated_responses, total_latency_seconds=total_latency)

    except json.JSONDecodeError as json_err:
        logger.error(f"JSON decoding error: {json_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse response from LightRAG.")
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")



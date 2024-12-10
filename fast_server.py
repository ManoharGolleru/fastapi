from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import json
import logging
import uvicorn
import csv
from concurrent.futures import ThreadPoolExecutor
import requests
import asyncio
from datetime import datetime
from dateutil.parser import parse
import os
import pytz

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Main Server", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=10)

# Updated API URL to point to the new LightRAG API
api_url = "http://localhost:7000/generate_response_informed"
rasp_pi_api_url = "https://humane-marmot-entirely.ngrok-free.app/"
headers = {"Content-Type": "application/json"}

# Directory to store session CSV files
session_csv_dir = "session_csv_files"
os.makedirs(session_csv_dir, exist_ok=True)

# In-memory history of the conversation (last 3 prompts and responses)
conversation_history = []
full_conversation_history = []
csv_file_path = None
time_responses_sent = None
time_chosen_response_received = None

# Eastern Time zone with DST handling
ET = pytz.timezone('US/Eastern')

# Generate a unique filename for each session
def generate_csv_filename():
    timestamp = datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    return os.path.join(session_csv_dir, f"conversation_history_{timestamp}.csv")

def get_speech_to_text():
    try:
        response = requests.get(f'{rasp_pi_api_url}/get_audio_transcription', timeout=10)
        response.raise_for_status()
        data_json = response.json()
        return data_json
    except requests.RequestException as e:
        logger.error(f"Error fetching speech-to-text: {e}")
        return {}

def check_last_entry(history):
    if history and history[-1][1] is None:
        logger.warning("Incomplete entry found in conversation history.")
        return handle_incomplete_entry(history)
    return None

def handle_incomplete_entry(history):
    incomplete_entry = history.pop()
    logger.info(f"Removed incomplete entry: {incomplete_entry[0]}")
    return f"Didn't choose a response; removed: {incomplete_entry[0]}"

def initialize_csv_file(path):
    try:
        with open(path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'index', 'date_time', 'prompt', 'history', 'responses', 
                'chosen_response', 'response_type', 'server_to_pi_latency', 
                'pi_processing_latency', 'pi_to_server_latency', 
                'api_latency', 'chosen_response_latency'
            ])
        logger.info(f"Initialized CSV file at {path}")
    except Exception as e:
        logger.error(f"Failed to create CSV: {e}")

def append_to_csv_file(path, entry):
    try:
        with open(path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(entry)
        logger.info(f"Appended entry to CSV at {path}")
    except Exception as e:
        logger.error(f"Failed to append to CSV: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global csv_file_path, conversation_history, full_conversation_history, time_responses_sent
    csv_file_path = generate_csv_filename()
    conversation_history = []
    full_conversation_history = []
    initialize_csv_file(csv_file_path)

    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                time_received_osdpi = datetime.now(ET)
                logger.info(f"Data received from OS-DPI at {time_received_osdpi}")

            try:
                data_json = json.loads(data)
                state = data_json.get("state", {})
                prefix = state.get("$prefix", "")
                emotion = state.get("$Style", "")

                if prefix == 'prompt':
                    incomplete_message = check_last_entry(conversation_history)
                    
                    time_server_sent_to_rasp_pi = datetime.now(ET)
                    rasp_pi_data = get_speech_to_text()
                    time_server_received_from_rasp_pi = datetime.now(ET)
                    
                    prompt = rasp_pi_data.get('text', '')
                    if not prompt:
                        logger.error("No prompt text received from RPi API.")
                        await websocket.send_text(json.dumps({'error': 'No prompt text received.'}))
                        continue

                    # Parse timestamps; fallback to current time if not provided
                    time_rasp_pi_received_from_server = parse(
                        rasp_pi_data.get('time_received', datetime.now().isoformat())
                    ).astimezone(ET)
                    time_rasp_pi_sent_to_server = parse(
                        rasp_pi_data.get('time_processed', datetime.now().isoformat())
                    ).astimezone(ET)
                    pi_processing_latency = rasp_pi_data.get('total_time', 0)

                    server_to_pi_latency = (time_rasp_pi_received_from_server - time_server_sent_to_rasp_pi).total_seconds()
                    pi_to_server_latency = (time_server_received_from_rasp_pi - time_rasp_pi_sent_to_server).total_seconds()

                    logger.info(f"Prompt received: {prompt}")
                    message = json.dumps({'state': {"$Display": prompt}})
                    
                    logger.debug(f"Latencies - Server to Pi: {server_to_pi_latency}, Pi Processing: {pi_processing_latency}, Pi to Server: {pi_to_server_latency}")

                    await websocket.send_text(message)

                    if prompt:
                        logger.info(f"Sending prompt to API with Emotion: {emotion}")
                        loop = asyncio.get_running_loop()

                        # Prepare parameters for the API
                        number_of_responses = 2  # Example: 2 responses
                        response_types = ["positive", "negative"]  # Example types
                        search_mode = "hybrid"  # Example search mode

                        # Optionally, retrieve these parameters from data_json if provided
                        # number_of_responses = data_json.get("number_of_responses", 2)
                        # response_types = data_json.get("response_types", ["positive", "negative"])
                        # search_mode = data_json.get("search_mode", "hybrid")

                        api_request_start_time = datetime.now(ET)
                        response = await loop.run_in_executor(
                            executor, 
                            send_to_api_sync, 
                            prompt, 
                            number_of_responses, 
                            response_types, 
                            search_mode
                        )
                        api_request_end_time = datetime.now(ET)

                        api_latency = (api_request_end_time - api_request_start_time).total_seconds()

                        responses_list = response.get('responses', [])

                        # Structure the responses to include type and text
                        responses_dict = {
                            f"response{i+1}": {
                                "type": resp['response_type'],
                                "text": resp['response_text']
                            } for i, resp in enumerate(responses_list)
                        }
                        responses_dict['Display'] = prompt
                        if incomplete_message:
                            responses_dict['warning'] = incomplete_message

                        logger.debug(f"API responses: {responses_dict}")
                        time_responses_sent = datetime.now(ET)
                        await websocket.send_text(json.dumps(responses_dict))

                        # Update conversation history
                        update_history(
                            conversation_history, 
                            prompt, 
                            None, 
                            responses_list, 
                            full_conversation_history, 
                            emotion, 
                            server_to_pi_latency, 
                            pi_processing_latency, 
                            pi_to_server_latency, 
                            api_latency
                        )
                    else:
                        logger.error("No prompt found in the received data.")
                
                elif prefix == 'Chosen':
                    chosen_response = state.get("$socket", "")
                    time_chosen_response_received = datetime.now(ET)
                    chosen_response_latency = (time_chosen_response_received - time_responses_sent).total_seconds()

                    if chosen_response:
                        logger.info(f"Received chosen response: {chosen_response}")
                        if conversation_history and conversation_history[-1][1] is None:
                            conversation_history[-1] = (conversation_history[-1][0], chosen_response)
                            update_full_history(full_conversation_history, conversation_history[-1], chosen_response)
                            timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
                            append_to_csv_file(csv_file_path, (
                                len(conversation_history),
                                timestamp,
                                conversation_history[-1][0],
                                full_conversation_history[-1][1],
                                chosen_response,
                                full_conversation_history[-1][4],
                                conversation_history[-1][2],  # server_to_pi_latency
                                conversation_history[-1][3],  # pi_processing_latency
                                conversation_history[-1][4],  # pi_to_server_latency
                                conversation_history[-1][5],  # api_latency
                                chosen_response_latency
                            ))
                        else:
                            logger.error("Chosen response received without a corresponding prompt.")
                    else:
                        logger.error("No chosen response found in the received data.")
                
                elif prefix == 'new_conv':
                    logger.info("Received new_conv prefix, clearing conversation history and starting new conversation.")
                    conversation_history.clear()
                    full_conversation_history.clear()
                    if csv_file_path:
                        append_to_csv_file(csv_file_path, ("", "", "", "", "", "", "", "", "", "", ""))
                
                else:
                    logger.error(f"Unexpected prefix value: {prefix}")

            except json.JSONDecodeError:
                logger.error("Invalid JSON received.")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")

def update_history(history, partner_prompt, user_response, model_responses, full_history, emotion, server_to_pi_latency, pi_processing_latency, pi_to_server_latency, api_latency):
    history_snapshot = history[-3:]
    while len(history) > 3:
        history.pop(0)
    history.append((partner_prompt, user_response, server_to_pi_latency, pi_processing_latency, pi_to_server_latency, api_latency))
    if model_responses is not None:
        full_history.append((partner_prompt, model_responses, user_response, history_snapshot, emotion))

def update_full_history(full_history, last_convo_pair, chosen_response):
    for index, (partner_prompt, model_responses, user_response, history_snapshot, emotion) in enumerate(full_history):
        if partner_prompt == last_convo_pair[0] and user_response is None:
            full_history[index] = (partner_prompt, model_responses, chosen_response, history_snapshot, emotion)
            break

@app.get("/download_csv")
async def download_csv():
    global csv_file_path
    try:
        if csv_file_path and os.path.exists(csv_file_path):
            return FileResponse(csv_file_path, media_type='text/csv', filename=os.path.basename(csv_file_path))
        else:
            logger.error("CSV file does not exist.")
            return {"error": "CSV file does not exist."}
    except Exception as e:
        logger.error(f"Failed to generate CSV: {e}")
        return {"error": f"Failed to generate CSV: {e}"}

def send_to_api_sync(prompt, number_of_responses, response_types, search_mode):
    try:
        payload = {
            'prompt': prompt,
            'number_of_responses': number_of_responses,
            'response_types': response_types,
            'search_mode': search_mode
        }
        logger.info(f"Sending payload to API: {payload}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        logger.info(f"Received response from API: {response.json()}")
        return response.json()
    except requests.HTTPError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return {"responses": []}
    except requests.RequestException as e:
        logger.error(f"Error sending request to API: {e}")
        return {"responses": []}

if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=5678, log_level="info")
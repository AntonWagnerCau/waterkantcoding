"""
SpotAgent - Voice-controlled agent for Boston Dynamics Spot robot
"""
import os
import time
import asyncio
import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import random
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Load environment variables
load_dotenv()

from audio_processor import AudioProcessor
from spot_controller import SpotController
from prompt_logger import PromptLogger
from llm_processor import LLMProcessor
from perception_logger import PerceptionLogger
from action_logger import ActionLogger

# --- FastAPI & WebSocket Setup ---

app = FastAPI()

# Add CORS middleware to allow frontend dev server to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
PORT = 5173

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        async with self.lock:
            await websocket.accept()
            self.active_connections.append(websocket)
            print(f"New connection established. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients, removing any that fail."""
        serialized_message = json.dumps(message)

        async with self.lock:
            connections_to_remove = []
            # Use enumerate for easier debugging of which connection failed
            for i, conn in enumerate(self.active_connections):
                try:
                    # Use send_text with the pre-serialized message
                    await conn.send_text(serialized_message)
                except Exception as e:
                    # Log specific connection index and error - KEEP basic error
                    print(f"[WS Broadcast] Error sending to client #{i}: {e}")
                    connections_to_remove.append(conn)
            
            # Remove any failed connections
            for conn in connections_to_remove:
                try:
                    self.active_connections.remove(conn)
                    # Keep basic removal log
                    print(f"[WS Broadcast] Removed closed connection index #{i}, {len(self.active_connections)} active connections remaining")
                except ValueError:
                    pass  # Already removed

manager = ConnectionManager()

# Shared state dictionary
agent_state: Dict[str, Any] = {
    "status": "Initializing",
    "current_task_prompt": None,
    "last_thought": None,
    "last_action": None,
    "last_action_params": None,
    "task_complete": False,
    "task_success": None,
    "task_reason": None,
    "vision_analysis": None,
    "odometry": None,
    "object_detection": {
        "status": "pending",
        "objects": [],
        "object_count": 0,
        "base64_images": {},
        "error": None
    },
}

async def update_state_and_broadcast(update_data: Dict[str, Any]):
    """Updates the global state and broadcasts it to all connected clients."""
    global agent_state
    # print(f"Updating state with: {update_data}") # Debug
    agent_state.update(update_data)
    await manager.broadcast(agent_state)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_json(agent_state)
        while True:
            await asyncio.sleep(1) # Prevent tight loop if not receiving
    except WebSocketDisconnect:
        print("Client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        manager.disconnect(websocket)

# --- Run FastAPI in a separate thread ---

server_thread = None
server_should_run = True
server_loop = None  # Add global variable to store the event loop

def run_server():
    global server_loop, server_should_run
    # Need to get or create an event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_loop = loop  # Store the server's event loop

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)

    async def main_loop():
        try:
            await server.serve()
        except asyncio.CancelledError:
            print("Server task cancelled.")
        finally:
            print("Server shutdown complete.")

    server_task = loop.create_task(main_loop())
    
    global server_should_run
    while server_should_run:
        loop.run_until_complete(asyncio.sleep(0.1))
        
    print("Attempting graceful server shutdown...")
    server_task.cancel()
    loop.run_until_complete(server_task)
    loop.close()
    print("Server thread finished.")

# Mount static files (assuming frontend build is in 'frontend/dist')
frontend_dist_path = os.path.join("frontend", "dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static")
else:
    print(f"Warning: Frontend directory '{frontend_dist_path}' not found. Static file serving disabled.")
    @app.get("/")
    async def read_root():
        return {"message": f"SpotAgent backend running. Frontend not found at {frontend_dist_path}."}

# --- Main Application Logic ---

def main():
    """Main application function"""
    print("Starting SpotAgent...")
    
    # Start the web server in a separate thread
    global server_thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print(f"Web server started on http://0.0.0.0:{PORT}")
    
    # Need to get the main thread's event loop or create one if needed
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError: # 'RuntimeError: Cannot run the event loop while another loop is running'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def schedule_state_update(update_data: Dict[str, Any]):
        """Schedules the async state update from the synchronous main thread."""
        global server_loop
        if (server_thread and 
            server_thread.is_alive() and 
            server_loop and 
            not server_loop.is_closed()):
            
            # Use the server's event loop to schedule the coroutine
            asyncio.run_coroutine_threadsafe(
                update_state_and_broadcast(update_data),
                server_loop
            )
        else:
            print("Warning: Server thread not running, cannot update state")

    # Initial state update
    schedule_state_update({"status": "Connecting to Spot..."})

    spot_controller = SpotController()
    # Connect to Spot robot
    print("Connecting to Spot robot...")
    if os.getenv("SPOT_IP"):
        spot_connected = spot_controller.connect()
        if spot_connected:
            print("Connected to Spot successfully!")
            schedule_state_update({"status": "Connected to Spot"})
        else:
            print("Failed to connect to Spot. Running in simulation mode.")
            schedule_state_update({"status": "Simulation Mode (Connection Failed)"})
    else:
        print("No Spot robot configuration found. Running in simulation mode.")
        spot_connected = False
        schedule_state_update({"status": "Simulation Mode (No Config)"})
    
    # Initialize the prompt logger
    prompt_logger = PromptLogger()
    
    # Initialize the LLM processor with the prompt logger
    llm_processor = LLMProcessor(prompt_logger=prompt_logger, spot_controller=spot_controller)
    audio_processor = AudioProcessor()
    
    # Initialize and start the perception logger
    # Pass the state update function to the logger if it needs to update state directly
    perception_logger = PerceptionLogger(spot_controller, state_update_callback=schedule_state_update) 
    perception_logger.start()
    
    # Initialize the action logger
    action_logger = ActionLogger()
    
    # Main loop
    try:
        schedule_state_update({"status": "Idle - Ready for command"}) 
        while True:
            # Prompt the user for text input
            print("Enter your command (or 'exit' to quit):")
            text = input("> ").strip()
            
            if not text:
                continue  # Skip empty inputs
            if text.lower() in {"exit", "quit"}:
                print("Exiting SpotAgent.")
                break

            if text:
                print(f"Transcription received: {text}")
                schedule_state_update({"status": "Processing Command", "current_task_prompt": text})
                # Reset conversation for new task
                llm_processor.reset_conversation()
                
                # Initialize task state
                task_complete = False
                task_data = None
                task_success = None
                task_reason = None
                
                # Start new task log
                task_log_file = action_logger.start_new_task(text)
                
                # Task execution loop - continue until task is marked complete
                while not task_complete:
                    # Get current action log to provide as context
                    current_action_log = action_logger.get_current_task_log()
                    
                    schedule_state_update({"status": "Thinking...", "current_task_prompt": text if task_data is None else "Continuing task..."})
                    # Process current state with LLM
                    action_data = llm_processor.process_command(
                        text if task_data is None else "Continue task execution or report task complete", 
                        task_data,
                        current_action_log
                    )
                    
                    if not action_data:
                        print("Failed to get a valid response from LLM")
                        action_logger.log_task_completion(False, "Failed to get valid LLM response")
                        task_complete = True
                        task_success = False
                        task_reason = "LLM Error"
                        schedule_state_update({"status": "Error", "task_complete": True, "task_success": False, "task_reason": task_reason})
                        break
                    
                    # Extract task status
                    task_status = action_data.get('task_status', {})
                    thought = action_data.get('thought', '')
                    action = action_data.get('action', '')
                    params = action_data.get('parameters', {})
                    
                    # Log the action and thought
                    action_logger.log_action(thought, action, params, task_status)
                    
                    task_complete = task_status.get('complete', False) or action == 'task_complete' or action_data["action"] is None or action_data.get('action', '').lower() in ['', 'none', 'stop', 'exit', 'quit', 'null']
                    task_success = task_status.get('success', None) # Success might only be set on completion
                    task_reason = task_status.get('reason', None)
                    
                    # Update state for UI
                    schedule_state_update({
                        "status": f"Executing: {action}",
                        "last_thought": thought,
                        "last_action": action,
                        "last_action_params": params,
                        "task_complete": task_complete,
                        "task_success": task_success, # May be None until task ends
                        "task_reason": task_reason, # May be None until task ends
                    })
                    
                    print(f"Thought: {thought}")
                    print(f"Action: {action}")
                    print(f"Parameters: {params}")
                    print(f"Task status: {task_status}")
                    
                    # Execute command based on action
                    # Store results for next iteration
                    task_data = {"last_action": action, "last_result": None}
                    action_result = None # Store specific results for state update
                    
                    if action == 'relative_move':
                        result = spot_controller.relative_move(
                            params.get('x', 0) or 0, 
                            params.get('y', 0) or 0
                        )
                        action_result = {"success": result}
                    
                    elif action == 'turn':
                        result = spot_controller.turn(params.get('degrees', 90))
                        action_result = {"success": result}
                    
                    elif action == 'sit':
                        result = spot_controller.sit()
                        action_result = {"success": result}
                    
                    elif action == 'stand':
                        result = spot_controller.stand()
                        action_result = {"success": result}

                    elif action == 'tilt':
                        result = spot_controller.tilt()
                        action_result = {"success": result}
                    
                    elif action == 'task_complete':
                        # The LLM has decided the task is complete
                        task_complete = True
                        task_success = params.get('success', True)
                        task_reason = params.get('reason', "Task completed successfully")
                        print(f"Task completed: {task_success}, Reason: {task_reason}")
                        
                        # Log final task completion status
                        action_logger.log_task_completion(task_success, task_reason)
                        # Update state explicitly here as loop will exit
                        schedule_state_update({
                            "status": "Task Complete",
                            "task_complete": True,
                            "task_success": task_success,
                            "task_reason": task_reason,
                        })
                    
                    else:
                        print(f"Unknown action: {action}")
                        action_result = {"error": f"Unknown action: {action}"}
                    
                    # Update task_data for next LLM iteration
                    task_data["last_result"] = action_result
                    
                    # Pause briefly between actions only if task is not complete
                    if not task_complete:
                        time.sleep(1)
                
                # After loop finishes (task complete)
                print("Task execution complete.")
                final_status = agent_state.get("status", "Task Ended") # Keep status if already set (e.g., Error)
                final_success = agent_state.get("task_success")
                final_reason = agent_state.get("task_reason")
                
                # If task_complete wasn't triggered by task_complete action, log and update state
                if not action == 'task_complete':
                    # If success/reason weren't set by the LLM, assume success
                    if task_success is None:
                        task_success = True 
                        task_reason = "Task finished without explicit completion status."
                    action_logger.log_task_completion(task_success, task_reason)
                    final_success = task_success
                    final_reason = task_reason
                    final_status = "Task Complete" if task_success else "Task Failed"

                schedule_state_update({ 
                    "status": final_status,
                    "task_complete": True, 
                    "task_success": final_success,
                    "task_reason": final_reason
                }) 
            else:
                schedule_state_update({"status": "Idle - Ready for command"}) # Update status if no text heard
                time.sleep(0.5)  # Brief pause before checking again
            
            # Slight pause before checking transcription again
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
        schedule_state_update({"status": "Shutting down..."})
    finally:
        # Clean up
        print("Stopping perception logger...")
        perception_logger.stop()
        if spot_connected:
            print("Disconnecting from Spot...")
            spot_controller.disconnect()
        
        # Signal the server thread to stop
        print("Stopping web server...")
        global server_should_run
        server_should_run = False
        if server_thread:
            server_thread.join() # Wait for the server thread to finish
            
        print("SpotAgent terminated")

if __name__ == "__main__":
    main() 
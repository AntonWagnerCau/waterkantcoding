"""
SpotAgent - Voice-controlled agent for Boston Dynamics Spot robot
"""
import sys
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
import socket


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
PORT_SIM = 65432

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


# Mount static files (assuming frontend build is in 'frontend/dist')
frontend_dist_path = os.path.join("frontend", "dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static")
else:
    print(f"Warning: Frontend directory '{frontend_dist_path}' not found. Static file serving disabled.")
    @app.get("/")
    async def read_root():
        return {"message": f"SpotAgent backend running. Frontend not found at {frontend_dist_path}."}


class SpotAgent():
    def __init__(self, spot_controller = None, isSim=False):
        self.action_logger = ActionLogger()
        self.spot_controller = spot_controller if spot_controller else SpotController()
        self.running = True
        self.spot_connected = False
        self.isSim = isSim

    def start(self):
        print("Starting SpotAgent...")

        # Start the web server in a separate thread
        global server_thread
        server_thread = threading.Thread(target=self.run_server, daemon=True)
        server_thread.start()
        print(f"Web server started on http://0.0.0.0:{PORT}")

        if (self.isSim):
            simulation_thread = threading.Thread(target=self.tcp_server_loop, daemon=True)
            simulation_thread.start()
            print(f"Simulation server started on http://127.0.0.1:{PORT_SIM}")

        # Need to get the main thread's event loop or create one if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError: # 'RuntimeError: Cannot run the event loop while another loop is running'
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.schedule_state_update({"status": "Connecting to Spot..."})

        # Connect to Spot robot
        print("Connecting to Spot robot...")
        if os.getenv("SPOT_IP"):
            self.spot_connected = self.spot_controller.connect()
            if self.spot_connected:
                print("Connected to Spot successfully!")
                self.schedule_state_update({"status": "Connected to Spot"})
            else:
                print("Failed to connect to Spot. Running in simulation mode.")
                self.schedule_state_update({"status": "Simulation Mode (Connection Failed)"})
        else:
            print("No Spot robot configuration found. Running in simulation mode.")
            self.spot_connected = False
            self.schedule_state_update({"status": "Simulation Mode (No Config)"})
        

        # Initialize the prompt logger
        prompt_logger = PromptLogger()
        
        # Initialize the LLM processor with the prompt logger
        self.llm_processor = LLMProcessor(prompt_logger=prompt_logger, spot_controller=self.spot_controller)
        self.audio_processor = AudioProcessor()
        
        # Initialize and start the perception logger
        # Pass the state update function to the logger if it needs to update state directly
        self.perception_logger = PerceptionLogger(self.spot_controller, state_update_callback=self.schedule_state_update) 
        self.perception_logger.start()
        
        # Initialize the action logger
        self.action_logger = ActionLogger()
        print("Initialized Spot Agent")

    def listen(self):
        try:
            self.schedule_state_update({"status": "Idle - Ready for command"}) 
            while True:
                # Start recording for initial command if not already recording
                if not self.audio_processor.is_recording:
                    print("Ready for next command. Speak now...")
                    self.schedule_state_update({"status": "Listening..."})
                    self.audio_processor.start_recording()

                # Check for transcription periodically
                self.schedule_state_update({"status": "Listening..."})
                text = self.audio_processor.get_transcription()
                self.handle_llm_input(text)

                # Slight pause before checking transcription again
                time.sleep(0.1)

            
        except KeyboardInterrupt:
            print("Interrupted by user")
            self.schedule_state_update({"status": "Shutting down..."})

    def tcp_server_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(("127.0.0.1", PORT_SIM))
            server_socket.listen()
            print("[Robot Server] Waiting for incoming connections...")
            
            while self.running:
                conn, addr = server_socket.accept()
                print(f"[Robot Server] Connected by {addr}")
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()


    def run_server(self):
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

    
    def handle_client(self, conn):
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                text = data.decode('utf-8').strip()
                print(f"[Robot Server] Received command: {text}")
                if text.lower() in ["exit", "quit"]:
                    print("[Robot Server] Received shutdown command.")
                    self.running = False
                    break

                if text:
                    print(f"[Robot Server] Sending command to robot...")
                    self.handle_llm_input(text)
                    print("[Robot Server] Command execution finished.")
                    conn.sendall(b"Command executed successfully.\n")
                else:
                    conn.sendall(b"Empty command received.\n")
        
    async def update_state_and_broadcast(self, update_data: Dict[str, Any]):
        """Updates the global state and broadcasts it to all connected clients."""
        global agent_state
        # print(f"Updating state with: {update_data}") # Debug
        agent_state.update(update_data)
        await manager.broadcast(agent_state)

    def schedule_state_update(self, update_data: Dict[str, Any]):
        """Schedules the async state update from the synchronous main thread."""
        global server_loop, server_thread

        if (server_thread and 
            server_thread.is_alive() and 
            server_loop and 
            not server_loop.is_closed()):
            
            # Use the server's event loop to schedule the coroutine
            asyncio.run_coroutine_threadsafe(
                self.update_state_and_broadcast(update_data),
                server_loop
            )
        else:
            print("Warning: Server thread not running, cannot update state")


    def handle_llm_input(self, text):
        if text:
            self.schedule_state_update({"status": "Processing Command", "current_task_prompt": text})
            # Reset conversation for new task
            self.llm_processor.reset_conversation()
            
            # Initialize task state
            task_complete = False
            task_data = None
            task_success = None
            task_reason = None
            
            # Start new task log
            task_log_file = self.action_logger.start_new_task(text)
            
            # Task execution loop - continue until task is marked complete
            while not task_complete:
                # Get current action log to provide as context
                current_action_log = self.action_logger.get_current_task_log()
                
                self.schedule_state_update({"status": "Thinking...", "current_task_prompt": text if task_data is None else "Continuing task..."})
                # Process current state with LLM
                action_data = self.llm_processor.process_command(
                    text if task_data is None else "Continue task execution or report task complete", 
                    task_data,
                    current_action_log
                )
                
                if not action_data:
                    print("Failed to get a valid response from LLM")
                    self.action_logger.log_task_completion(False, "Failed to get valid LLM response")
                    task_complete = True
                    task_success = False
                    task_reason = "LLM Error"
                    self.schedule_state_update({"status": "Error", "task_complete": True, "task_success": False, "task_reason": task_reason})
                    break
                
                # Extract task status
                task_status = action_data.get('task_status', {})
                thought = action_data.get('thought', '')
                action = action_data.get('action', '')
                params = action_data.get('parameters', {})
                
                # Log the action and thought
                self.action_logger.log_action(thought, action, params, task_status)
                
                task_complete = task_status.get('complete', False) or action == 'task_complete' or action_data["action"] is None or action_data.get('action', '').lower() in ['', 'none', 'stop', 'exit', 'quit', 'null']
                task_success = task_status.get('success', None) # Success might only be set on completion
                task_reason = task_status.get('reason', None)
                
                # Update state for UI
                self.schedule_state_update({
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
                    result = self.spot_controller.relative_move(
                        params.get('x', 0) or 0, 
                        params.get('y', 0) or 0
                    )
                    action_result = {"success": result}
                
                elif action == 'turn':
                    result = self.spot_controller.turn(params.get('degrees', 90))
                    action_result = {"success": result}
                
                elif action == 'sit':
                    result = self.spot_controller.sit()
                    action_result = {"success": result}
                
                elif action == 'stand':
                    result = self.spot_controller.stand()
                    action_result = {"success": result}
                
                elif action == 'task_complete':
                    # The LLM has decided the task is complete
                    task_complete = True
                    task_success = params.get('success', True)
                    task_reason = params.get('reason', "Task completed successfully")
                    print(f"Task completed: {task_success}, Reason: {task_reason}")
                    
                    # Log final task completion status
                    self.action_logger.log_task_completion(task_success, task_reason)
                    # Update state explicitly here as loop will exit
                    self.schedule_state_update({
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
                self.action_logger.log_task_completion(task_success, task_reason)
                final_success = task_success
                final_reason = task_reason
                final_status = "Task Complete" if task_success else "Task Failed"

            self.schedule_state_update({ 
                "status": final_status,
                "task_complete": True, 
                "task_success": final_success,
                "task_reason": final_reason
            }) 
        else:
            self.schedule_state_update({"status": "Idle - Ready for command"}) # Update status if no text heard
            time.sleep(0.5)

    def clean_up(self):
        print("Stopping perception logger...")
        self.perception_logger.stop()
        if self.spot_connected:
            print("Disconnecting from Spot...")
            self.spot_controller.disconnect()
        
        # Signal the server thread to stop
        print("Stopping web server...")
        global server_should_run, server_thread
        server_should_run = False
        if server_thread:
            server_thread.join() # Wait for the server thread to finish
            
        print("SpotAgent terminated")


# --- Main Application Logic ---

def main():
    """Main application function"""
    print("Starting Main Thread...")
    spot_agent = SpotAgent()

    print("Starting the services")
    spot_agent.start()

    print("Starting main loop")
    spot_agent.listen()

    print("Starting clean up")
    spot_agent.clean_up()

if __name__ == "__main__":
    main() 

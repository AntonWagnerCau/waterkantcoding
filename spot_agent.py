"""
SpotAgent - Voice-controlled agent for Boston Dynamics Spot robot
"""
import os
import time
import json
import threading
from dotenv import load_dotenv
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Load environment variables
load_dotenv()

from audio_processor import AudioProcessor
from spot_controller import SpotController
from prompt_logger import PromptLogger
from llm_processor import LLMProcessor
from perception_logger import PerceptionLogger
from action_logger import ActionLogger

def main():
    """Main application function"""
    print("Starting SpotAgent...")
    audio_processor = AudioProcessor()
    
    # Initialize the prompt logger
    prompt_logger = PromptLogger()
    
    # Initialize the LLM processor with the prompt logger
    llm_processor = LLMProcessor(prompt_logger=prompt_logger)
    
    spot_controller = SpotController()
    
    # Connect to Spot robot
    print("Connecting to Spot robot...")
    if os.getenv("SPOT_IP"):
        spot_connected = spot_controller.connect()
        if spot_connected:
            print("Connected to Spot successfully!")
        else:
            print("Failed to connect to Spot. Running in simulation mode.")
    else:
        print("No Spot robot configuration found. Running in simulation mode.")
        spot_connected = False
    
    # Initialize and start the perception logger
    perception_logger = PerceptionLogger(spot_controller)
    perception_logger.start()
    
    # Initialize the action logger
    action_logger = ActionLogger()
    
    # Main loop
    try:
        while True:
            # Start recording for initial command
            audio_processor.start_recording()
            
            # Wait for recording to finish
            while audio_processor.is_recording:
                time.sleep(0.1)
            
            # Get transcription
            text = audio_processor.get_transcription()
            
            # Exit if asked
            if text and any(keyword in text.lower() for keyword in ["exit", "quit", "stop"]):
                print("Exiting application...")
                break
            
            # Process with LLM and start task execution loop
            if text:
                # Reset conversation for new task
                llm_processor.reset_conversation()
                
                # Initialize task state
                task_complete = False
                task_data = None
                
                # Start new task log
                task_log_file = action_logger.start_new_task(text)
                
                # Task execution loop - continue until task is marked complete
                while not task_complete:
                    # Get current action log to provide as context
                    current_action_log = action_logger.get_current_task_log()
                    
                    # Process current state with LLM
                    action_data = llm_processor.process_command(
                        text if task_data is None else "Continue task execution", 
                        task_data,
                        current_action_log
                    )
                    
                    if not action_data:
                        print("Failed to get a valid response from LLM")
                        action_logger.log_task_completion(False, "Failed to get valid LLM response")
                        break
                    
                    # Extract task status
                    task_status = action_data.get('task_status', {})
                    thought = action_data.get('thought', '')
                    action = action_data.get('action', '')
                    params = action_data.get('parameters', {})
                    
                    # Log the action and thought
                    action_logger.log_action(thought, action, params, task_status)
                    
                    task_complete = task_status.get('complete', False) or task_status.get('success', False) or action_data["action"] is None or action_data.get('action', '').lower() in ['', 'none', 'stop', 'exit', 'quit', 'null']
                    
                    print(f"Thought: {thought}")
                    print(f"Action: {action}")
                    print(f"Parameters: {params}")
                    print(f"Task status: {task_status}")
                    
                    # Execute command based on action
                    # Store results for next iteration
                    task_data = {"last_action": action, "last_result": None}
                    
                    # Execute the action
                    if action == 'walk_forward' or action == 'relative_move' and params.get('forward_backward', 0) > 0:
                        result = spot_controller.walk_forward(params.get('forward_backward', 1.0))
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'walk_backward' or action == 'relative_move' and params.get('forward_backward', 0) < 0:
                        result = spot_controller.walk_backward(abs(params.get('forward_backward', 1.0)))
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'relative_move':
                        result = spot_controller.relative_move(
                            params.get('forward_backward', 0), 
                            params.get('left_right', 0)
                        )
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'turn':
                        result = spot_controller.turn(params.get('degrees', 90))
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'sit':
                        result = spot_controller.sit()
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'stand':
                        result = spot_controller.stand()
                        task_data["last_result"] = {"success": result}
                    
                    elif action == 'take_picture':
                        image_path = spot_controller.take_pictures()
                        task_data["last_result"] = {"success": image_path is not None, "image_path": image_path}
                    
                    elif action == 'get_odometry':
                        odometry = spot_controller.get_odometry()
                        task_data["last_result"] = odometry
                    
                    elif action == 'analyze_image':
                        vision_result = spot_controller.analyze_images()
                        task_data["last_result"] = vision_result
                    
                    elif action == 'get_odometry_logs':
                        seconds = params.get('seconds', 1)
                        logs = spot_controller.get_odometry_logs(seconds)
                        task_data["last_result"] = logs
                    
                    elif action == 'get_vision_logs':
                        seconds = params.get('seconds', 1)
                        logs = spot_controller.get_vision_logs(seconds)
                        task_data["last_result"] = logs
                        
                    elif action == 'task_complete':
                        # The LLM has decided the task is complete
                        task_complete = True
                        success = params.get('success', True)
                        reason = params.get('reason', "Task completed successfully")
                        print(f"Task completed: {success}, Reason: {reason}")
                        
                        # Log final task completion status
                        action_logger.log_task_completion(success, reason)
                    
                    else:
                        print(f"Unknown action: {action}")
                        task_data["last_result"] = {"error": f"Unknown action: {action}"}
                    
                    # Pause briefly between actions
                    time.sleep(1)
                
                print("\nTask execution complete. Ready for next command.")
                
                # If task_complete wasn't triggered by task_complete action, log it here
                if not action == 'task_complete':
                    action_logger.log_task_completion(True, "Task completed without explicit completion action")
            
            print("\nReady for next command. Speak now...")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        perception_logger.stop()
        if spot_connected:
            spot_controller.disconnect()
        print("SpotAgent terminated")

if __name__ == "__main__":
    main() 
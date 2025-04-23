# SpotAgent

A voice-controlled agent for Boston Dynamics Spot robot. This application uses completely free and open source components:

1. Captures voice input from your microphone
2. Converts speech to text using OpenAI's Whisper (runs offline)
3. Processes commands with a local LLM via Ollama/Google/Openrouter
4. Controls a Boston Dynamics Spot robot based on user voice commands

## Setup

Skip 1, 2, 6 if not running LLM locally
1. Install Ollama from https://ollama.com/

2. Pull a language model (Mistral recommended for balance of performance and quality):
   ```
   ollama pull mistral
   ```

3. No need to download any speech recognition models - the application will automatically download the Whisper model on first run (small download, ~150MB for base model).

4. Create a `.env` file with your settings:
   ```
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=mistral
   WHISPER_MODEL=base.en
   WHISPER_COMPUTE_TYPE=float32
   SPOT_IP=your_spot_robot_ip
   SPOT_USERNAME=your_spot_username
   SPOT_PASSWORD=your_spot_password
   ```

5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Start Ollama server (if not already running):
   ```
   ollama serve
   ```

7. Run the application:
   ```
   python spot_agent.py
   ```

## Usage

Speak commands into your microphone to control the Spot robot. Examples:
- "Walk forward 2 meters"
- "Turn right 90 degrees"
- "Sit down"
- "Stand up"

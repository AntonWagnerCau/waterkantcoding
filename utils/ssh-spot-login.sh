#!/bin/bash

# Load environment variables from .env file
set -a
source .env.example
set +a
# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed. Install it first."
    exit 1
fi
# Connect via SSH using sshpass
sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$SSH_HOST"
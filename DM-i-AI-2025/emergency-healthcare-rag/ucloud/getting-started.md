# UCloud Setup Guide

**Important**: GPU resources are shared with limited hours. Stop GPU jobs when not in use and limit your team to **1 GPU instance** at a time.

## Overview
This tutorial covers setting up FastAPI and Ollama (for local LLMs) on UCloud.

## 1. Accept Team Invite
1. Accept your team's UCloud invite link and login with university credentials
2. Share the invite link only with your teammates (not other participants)
3. You may need to click the invite link again after signup to join the team
4. Note your team name (e.g., TEAM31)

## 2. Start GPU Instance

1. Navigate to https://cloud.sdu.dk/app/applications
2. If you get message stating that you need to reconnect to DeiC, press it and press "Reconnect"
3. Click **Terminal**
4. Enter a job name
5. Set hours to 1-4
6. Keep nodes at 1
7. Choose machine type: **uc1-l4-1**  (do not select multi-GPU nodes)
8. Under "Select folders to use":
   - Click **Add folder**
   - Choose your team drive (e.g., TEAM31)
   - Files in the team drive persist between runs (50GB storage). All files outside the drive folder will be deleted when the job finishes.
9. Click **Submit** and wait for job to start
10. Click **Open terminal**
11. Team drive location: `/work/TEAM31`
12. Verify GPU: `nvidia-smi`

## 3. Setup Ollama

**Note**: Save all files, programs, and data in the team drive to persist between jobs. Files outside the team drive are deleted when jobs finish.

1. (On the GPU instance) Navigate to team directory:
   ```bash
   cd /work/TEAM31
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/amboltio/DM-i-AI-2025
   ```

3. Install Ollama (this modified ollama installation script installs ollama in the working directory as opposed to /usr/bin):
   ```bash
   sh DM-i-AI-2025/emergency-healthcare-rag/ucloud/ollama-install.sh
   ```

4. Start a screen session:
   ```bash
   screen
   ```

5. Start Ollama server (this will start ollama on port 11434):
   ```bash
   OLLAMA_MODELS=/work/TEAM31/models /work/TEAM31/ollama/bin/ollama serve
   ```

6. Detach from screen: `Ctrl+A+D`
   - To Reconnect: `screen -r`

7. Download a model:
   ```bash
   /work/TEAM31/ollama/bin/ollama pull llama3.2:3b
   ```

8. Test Ollama in the terminal:
   ```bash
   /work/TEAM31/ollama/bin/ollama run llama3.2:3b
   ```
   (Press `Ctrl+D` to exit chat)

9. Verify server is running:
   ```bash
   curl -L localhost:11434
   ```

## 4. Setup FastAPI Server

1. (On the gpu instance) Navigate to project directory:
   ```bash
   cd /work/TEAM31/DM-i-AI-2025/emergency-healthcare-rag
   ```

2. Create and activate virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   python ucloud/api.py
   ```

5. Test the API:
   ```bash
   curl -X POST localhost:8000/predict   -H "Content-Type: application/json"   -d '{"statement": "constipation is a disease"}'
   ```
 
## 5. Setup Nginx (Public Access)

To expose the FastAPI endpoint to the internet:

1. Navigate to https://cloud.sdu.dk/app/applications
2. Search for "nginx" in the upper right corner
3. Select the nginx application
4. Configure the job:
   - Give it a name
   - Machine type: **uc1-gc1-1** (no GPU needed)
   - Duration: 1-4 hours

5. Optional parameters:
   - NGINX configuration: Select `TEAM31/DM-i-AI-2025/emergency-healthcare-rag/ucloud/nginx.conf`

6. Configure custom links to your application: 
   - Click "Add public link" (save this URL - it's your public endpoint)

7. Connect to other jobs: 
   - Click "Connect to job"
   - Hostname: `my-api`
   - Select the GPU job

6. Click **Submit** and wait for startup

7. Verify setup:
   - Check logs for any errors
   - Test the public link in your browser
   - Test from a different machine:
   ```bash
   curl -X POST https://public-link.cloud.aau.dk/predict  -H "Content-Type: application/json"   -d '{"statement": "constipation is a disease"}
   ```
### Restarting Jobs
To restart nginx or GPU terminal applications:
1. Go to https://cloud.sdu.dk/app/jobs
2. Double-click a finished job
3. Click "Run application again" 

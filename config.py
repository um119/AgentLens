"""
AgentLens Configuration
Settings and constants for the LLM discovery assistant
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Application Settings
APP_TITLE = "AgentLens"
APP_ICON = "🔍"
TAGLINE = "AI-Powered LLM Discovery Assistant for Agentic AI Workflows"

# Example queries for placeholder text
EXAMPLE_QUERIES = [
    "I'm building a marketing automation agent that creates campaigns, targets audiences, and generates reports.",
    "Best LLMs for a customer support agentic workflow with tool calling.",
    "Which models support function calling for coding agents?",
    "I need a model for a research agent that analyzes papers and extracts key insights.",
]

# Structured output template
LLM_OUTPUT_FIELDS = {
    "name": "LLM Name",
    "description": "Description",
    "parameters": "Parameters",
    "key_features": "Key Features",
    "tool_calling_support": "Tool/Function Calling Support",
}

# System prompt for LLM recommendations
SYSTEM_PROMPT = """You are an AI assistant that helps developers, researchers, and AI students discover
and compare Large Language Models (LLMs) specifically suited for agentic AI workflows.

Your task is to recommend the best LLMs based on the user's workflow description.
For each recommended LLM, provide:
1. Model Name (e.g., GPT-4o, Claude 3.5 Sonnet, Llama 3.1 70B)
2. Description (1-2 sentence summary of the model's strengths)
3. Parameters (model size in billions, e.g., 70B, 405B)
4. Key Features (relevant capabilities for the workflow)
5. Tool/Function Calling Support (Yes/No and details)

Focus on models that support tool calling, function execution, and structured outputs.

Format your response as a structured list with clear labels for each field."""
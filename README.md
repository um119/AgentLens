# AgentLens

AI-Powered LLM Discovery Assistant for Agentic AI Workflows

## Overview

AgentLens is an intelligent assistant that recommends the best LLMs for any agentic AI workflow — powered by OpenAI Responses API with web search and local Ollama models.

## Features

- **Web Search**: Uses OpenAI Responses API with real-time web search to find the latest LLM information
- **Local Models**: Integrates with Ollama for local open-source model comparison
- **Structured Output**: Presents recommendations with model name, description, parameters, key features, and tool calling support
- **Interactive UI**: Streamlit web interface with card layouts and comparison tables

## Tech Stack

- **AI Engine**: OpenAI Python SDK — Responses API with web_search tool
- **Local Models**: Ollama (llama3.2, mistral, qwen2.5, etc.)
- **Web UI**: Streamlit
- **Data Handling**: Pandas
- **Configuration**: python-dotenv

## Setup

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

Get your OpenAI API key from: https://platform.openai.com/

### 3. Install Ollama (Optional)

For local model support:

1. Download Ollama from https://ollama.ai/
2. Run `ollama serve`
3. Pull models: `ollama pull llama3.2`, `ollama pull mistral`

## Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## Usage

1. **Describe your workflow**: Enter a natural language description of your agentic AI project
2. **Search**: Click "Search LLMs" to get recommendations
3. **Review**: Explore recommended models with descriptions, features, and tool support
4. **Compare**: Use the comparison table to evaluate options
5. **Test Local Models**: Try Ollama models directly from the interface

### Example Queries

- "I'm building a marketing automation agent that creates campaigns"
- "Best LLMs for a customer support agentic workflow"
- "Which models support function calling for coding agents?"

## Project Structure

```
agentlens/
├── app.py              # Streamlit web application
├── agent_core.py        # LLM search logic with OpenAI
├── ollama_utils.py      # Ollama local model utilities
├── config.py           # Configuration and constants
├── .env                # Environment variables (API keys)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Learning Outcomes

- Configure and use OpenAI Python SDK with Responses API
- Implement real-time web search through AI APIs
- Design prompts for structured output parsing
- Integrate Ollama for local model management
- Build professional Streamlit web interfaces
- Compare cloud vs local LLM tradeoffs

## License

MIT

---

Built for NAVTTC | Corvit Systems | Digital Pakistan
AI (ML, DL) Engineering — Mid-Term Exam Project
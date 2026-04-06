"""
Agent Core - LLM Search Logic
Uses Ollama local models with web search for LLM recommendations
"""

from typing import List, Dict, Any, Optional
import ollama
import config
from web_search import get_llm_recommendations_search


class LLMAgent:
    """AI-powered LLM discovery assistant using Ollama + Web Search"""

    def __init__(self, model_name: str = "qwen3.5:cloud"):
        """Initialize the LLM agent with Ollama"""
        self.model = model_name

    def search_llms(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for LLMs suitable for the given agentic workflow

        Args:
            query: Natural language description of the agentic workflow

        Returns:
            List of dictionaries containing LLM recommendations
        """
        try:
            # Step 1: Search the web for latest LLM information
            search_results = get_llm_recommendations_search(query)

            # Step 2: Build the prompt with web search context
            prompt = self._build_prompt(query, search_results)

            # Step 3: Call Ollama for recommendations
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                temperature=0.7,
                stream=False,
            )

            # Step 4: Parse the response
            recommendations = self._parse_response(response, query)
            return recommendations

        except Exception as e:
            print(f"Error searching LLMs: {e}")
            return []

    def _build_prompt(self, query: str, search_results: Dict[str, Any]) -> str:
        """Build the prompt with web search context"""
        summary = search_results.get("summary", "")

        return f"""I am building an agentic AI workflow with the following description:
"{query}"

Based on your knowledge AND the following recent web search results about latest LLMs, recommend 5-8 best LLMs suitable for this workflow.

Web Search Results:
{summary}

For each recommended LLM, provide the following information in this exact format:
LLM_NAME: <model name>
DESCRIPTION: <1-2 sentence summary>
PARAMETERS: <model size (e.g., 7B, 70B, 405B)>
KEY_FEATURES: <comma-separated list of key capabilities>
TOOL_CALLING: <Yes/No and details about tool/function calling support>

Focus on models that:
- Support tool/function calling for agentic workflows
- Have good reasoning capabilities
- Are cost-effective for the use case
- Have sufficient context windows

Provide accurate, current information about these models."""

    def _parse_response(self, response, query: str) -> List[Dict[str, Any]]:
        """Parse the API response into structured LLM data"""
        recommendations = []

        try:
            # Get the response text
            response_text = response.response if hasattr(response, 'response') else str(response)

            # Parse the structured output
            llm_entries = self._extract_llm_entries(response_text)

            for entry in llm_entries:
                if self._is_valid_llm_entry(entry):
                    recommendations.append(entry)

        except Exception as e:
            print(f"Error parsing response: {e}")

        # If no structured parsing worked, use sample recommendations
        if not recommendations:
            recommendations = self.get_sample_recommendations(query)

        return recommendations

    def _extract_llm_entries(self, text: str) -> List[Dict[str, Any]]:
        """Extract LLM entries from response text"""
        entries = []
        lines = text.split('\n')
        current_entry = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("LLM_NAME:") or line.startswith("Model Name:") or line.startswith("- "):
                if current_entry:
                    entries.append(current_entry)
                # Extract name - remove "- " prefix if present
                name = line.lstrip("- ").split(":", 1)[-1].strip()
                if name:
                    current_entry = {"name": name}
                else:
                    current_entry = {}

            elif line.startswith("DESCRIPTION:") or line.startswith("Description:"):
                if "name" in current_entry:
                    current_entry["description"] = line.split(":", 1)[1].strip()

            elif line.startswith("PARAMETERS:") or line.startswith("Parameters:"):
                current_entry["parameters"] = line.split(":", 1)[1].strip()

            elif line.startswith("KEY_FEATURES:") or line.startswith("Features:") or line.startswith("Key Features:"):
                current_entry["key_features"] = line.split(":", 1)[1].strip()

            elif line.startswith("TOOL_CALLING:") or line.startswith("Tool Calling:") or line.startswith("Function Calling:"):
                current_entry["tool_calling_support"] = line.split(":", 1)[1].strip()

        # Add the last entry
        if current_entry and "name" in current_entry:
            entries.append(current_entry)

        return entries

    def _is_valid_llm_entry(self, entry: Dict[str, Any]) -> bool:
        """Check if the entry has the required fields"""
        required_fields = ["name"]
        return all(field in entry for field in required_fields)

    def get_sample_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """Get sample recommendations based on workflow type"""
        query_lower = query.lower()

        if "marketing" in query_lower or "campaign" in query_lower:
            return [
                {"name": "GPT-4o", "description": "Highly capable multimodal model optimized for complex reasoning and tool use.", "parameters": "~1.8T", "key_features": "Multimodal, 128K context, function calling", "tool_calling_support": "Yes - native function calling"},
                {"name": "Claude 3.5 Sonnet", "description": "Excellent for marketing content with strong reasoning and long context.", "parameters": "~175B", "key_features": "200K context, computer use, excellent writing", "tool_calling_support": "Yes - Tool Use API"},
                {"name": "Llama 3.1 70B", "description": "Open-source model with strong performance for automation.", "parameters": "70B", "key_features": "Open weights, fine-tunable, 128K context", "tool_calling_support": "Yes - via fine-tuning"}
            ]
        elif "coding" in query_lower or "developer" in query_lower:
            return [
                {"name": "GPT-4o", "description": "Best for coding with excellent code generation.", "parameters": "~1.8T", "key_features": "Code generation, debugging, multimodal", "tool_calling_support": "Yes - native function calling"},
                {"name": "Claude 3.5 Sonnet", "description": "Excellent for complex code analysis.", "parameters": "~175B", "key_features": "200K context, computer use", "tool_calling_support": "Yes - Tool Use API"},
                {"name": "DeepSeek Coder V2", "description": "Specialized coding model.", "parameters": "236B", "key_features": "Code-specific, 128K context", "tool_calling_support": "Yes"}
            ]
        elif "customer support" in query_lower or "support" in query_lower:
            return [
                {"name": "GPT-4o", "description": "Well-suited for customer service.", "parameters": "~1.8T", "key_features": "Natural conversation, tool use", "tool_calling_support": "Yes - function calling"},
                {"name": "Claude 3.5 Sonnet", "description": "Empathetic responses with context.", "parameters": "~175B", "key_features": "200K context, comprehension", "tool_calling_support": "Yes - Tool Use API"},
                {"name": "Mistral 7B", "description": "Fast, cost-effective for simple queries.", "parameters": "7B", "key_features": "Fast inference, low cost", "tool_calling_support": "Limited"}
            ]
        else:
            return [
                {"name": "GPT-4o", "description": "Versatile with strong reasoning.", "parameters": "~1.8T", "key_features": "Multimodal, 128K context", "tool_calling_support": "Yes"},
                {"name": "Claude 3.5 Sonnet", "description": "Excellent reasoning and long context.", "parameters": "~175B", "key_features": "200K context", "tool_calling_support": "Yes"},
                {"name": "Llama 3.1 70B", "description": "Open-source option.", "parameters": "70B", "key_features": "Open weights, 128K", "tool_calling_support": "Yes"},
                {"name": "Gemini 1.5 Pro", "description": "Massive context window.", "parameters": "Unknown", "key_features": "2M context, multimodal", "tool_calling_support": "Yes"}
            ]


DEFAULT_MODEL = "qwen3.5:cloud"


def search_llms_for_workflow(query: str, model_name: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    """Search for LLMs using Ollama + Web Search"""
    agent = LLMAgent(model_name=model_name)

    try:
        results = agent.search_llms(query)
        if results:
            return results
    except Exception as e:
        print(f"Ollama call failed: {e}")

    return agent.get_sample_recommendations(query)
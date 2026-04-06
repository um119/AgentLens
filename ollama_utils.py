"""
Ollama Utilities - Local Model Management
Handles listing and interacting with local Ollama models
"""

import ollama
from typing import List, Dict, Any, Optional
import config


class OllamaManager:
    """Manages local Ollama models"""

    def __init__(self, base_url: Optional[str] = None):
        """Initialize Ollama client"""
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.client = ollama

    def is_ollama_running(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = self.client.list()
            return True
        except Exception as e:
            print(f"Ollama not running: {e}")
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all locally installed Ollama models

        Returns:
            List of model information dictionaries
        """
        try:
            response = self.client.list()
            models = []

            if hasattr(response, 'models'):
                for model in response.models:
                    models.append({
                        "name": model.name,
                        "size": model.size,
                        "modified_at": model.modified_at,
                        "digest": model.digest if hasattr(model, 'digest') else None,
                    })
            elif isinstance(response, dict) and 'models' in response:
                for model in response['models']:
                    models.append({
                        "name": model.get('name', ''),
                        "size": model.get('size', 0),
                        "modified_at": model.get('modified_at', ''),
                        "digest": model.get('digest', ''),
                    })

            return models

        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary
        """
        try:
            response = self.client.show(model_name)
            info = {
                "name": response.name if hasattr(response, 'name') else model_name,
                "modelfile": response.modelfile if hasattr(response, 'modelfile') else "",
                "parameters": response.parameters if hasattr(response, 'parameters') else "",
                "template": response.template if hasattr(response, 'template') else "",
                "details": response.details if hasattr(response, 'details') else {},
            }
            return info
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None

    def test_model(self, model_name: str, prompt: str) -> Optional[str]:
        """
        Test a model with a prompt

        Args:
            model_name: Name of the model to test
            prompt: Test prompt

        Returns:
            Model response or None on error
        """
        try:
            response = self.client.generate(
                model=model_name,
                prompt=prompt,
                stream=False
            )
            return response.response if hasattr(response, 'response') else str(response)
        except Exception as e:
            print(f"Error testing model: {e}")
            return None

    def format_model_size(self, size_bytes: int) -> str:
        """Format model size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Get formatted metadata for a model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with formatted metadata
        """
        info = self.get_model_info(model_name)
        if not info:
            return {"name": model_name, "error": "Could not retrieve info"}

        # Extract parameter count from model name
        # Common patterns: llama3.2:7b, mistral:8x7b, qwen2.5:14b
        params = self._extract_parameters(model_name)

        return {
            "name": model_name,
            "parameters": params,
            "family": info.get("details", {}).get("family", "unknown"),
            "quantization": info.get("details", {}).get("quantization", "unknown"),
        }

    def _extract_parameters(self, model_name: str) -> str:
        """Extract parameter count from model name"""
        import re

        # Match patterns like 7b, 8x7b, 14b, 70b, 405b (case insensitive)
        patterns = [
            r'(\d+[bx]\d+b)',  # 8x7b
            r'(\d+b)',          # 7b, 70b
            r'(\d+\.\d+b)',     # 3.2b
        ]

        for pattern in patterns:
            match = re.search(pattern, model_name.lower())
            if match:
                return match.group(1).upper()

        return "Unknown"

    def get_suitable_models(self, workflow_type: str) -> List[Dict[str, Any]]:
        """
        Get models suitable for a specific workflow type

        Args:
            workflow_type: Type of workflow (coding, marketing, support, etc.)

        Returns:
            List of suitable models with recommendations
        """
        models = self.list_models()
        suitable = []

        workflow_lower = workflow_type.lower()

        for model in models:
            model_name = model["name"].lower()
            params = self._extract_parameters(model["name"])

            recommendation = {
                "name": model["name"],
                "size": self.format_model_size(model["size"]),
                "parameters": params,
                "suitability": self._assess_suitability(model_name, workflow_lower),
            }
            suitable.append(recommendation)

        return suitable

    def _assess_suitability(self, model_name: str, workflow_type: str) -> str:
        """Assess model suitability for a workflow"""
        # Simple heuristics based on model size and family
        if "coding" in workflow_type or "developer" in workflow_type:
            if "code" in model_name or "codellama" in model_name:
                return "Excellent"
            elif any(x in model_name for x in ["70b", "405b", "large"]):
                return "Good"
            elif any(x in model_name for x in ["7b", "8b", "small"]):
                return "Fair"
            return "Unknown"

        elif "marketing" in workflow_type or "content" in workflow_type:
            if any(x in model_name for x in ["70b", "405b", "large", "mixtral"]):
                return "Good"
            return "Fair"

        elif "support" in workflow_type or "chat" in workflow_type:
            if any(x in model_name for x in ["7b", "8b", "small", "mistral"]):
                return "Good (fast, cost-effective)"
            elif any(x in model_name for x in ["70b"]):
                return "Good (quality)"
            return "Fair"

        # Default assessment
        if any(x in model_name for x in ["70b", "405b", "large"]):
            return "Good"
        return "Fair"


def get_local_models() -> List[Dict[str, Any]]:
    """
    Convenience function to get local models

    Returns:
        List of local models
    """
    manager = OllamaManager()
    return manager.list_models()


def is_ollama_available() -> bool:
    """
    Check if Ollama is available

    Returns:
        True if Ollama is running
    """
    manager = OllamaManager()
    return manager.is_ollama_running()
"""
Web Search Module
Provides real-time web search for LLM information
"""

from typing import List, Dict, Any
from duckduckgo_search import DDGS
import config


class WebSearch:
    """Web search for latest LLM information"""

    def __init__(self):
        self.ddgs = DDGS()

    def search_llms(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web for LLM information

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with title, url, and snippet
        """
        try:
            results = self.ddgs.text(
                query,
                max_results=max_results
            )
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def search_llm_recommendations(self, workflow: str) -> Dict[str, Any]:
        """
        Search for LLM recommendations based on workflow

        Args:
            workflow: Description of the agentic workflow

        Returns:
            Dictionary with search results and summary
        """
        # Build search query
        search_query = f"best LLMs for {workflow} agentic AI workflow tool calling 2025"

        results = self.search_llms(search_query, max_results=10)

        # Extract relevant information
        search_data = {
            "query": workflow,
            "results": results,
            "summary": self._generate_summary(results)
        }

        return search_data

    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary from search results"""
        if not results:
            return "No search results found."

        summary_parts = []
        for i, result in enumerate(results[:5], 1):
            title = result.get("title", "")
            snippet = result.get("body", "")[:100]
            summary_parts.append(f"{i}. {title}: {snippet}...")

        return "\n".join(summary_parts)


def search_web(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function for web search

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        List of search results
    """
    searcher = WebSearch()
    return searcher.search_llms(query, max_results)


def get_llm_recommendations_search(workflow: str) -> Dict[str, Any]:
    """
    Get web search results for LLM recommendations

    Args:
        workflow: Workflow description

    Returns:
        Search results dictionary
    """
    searcher = WebSearch()
    return searcher.search_llm_recommendations(workflow)
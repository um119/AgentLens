"""
Web Search Module
Provides real-time web search for LLM information
"""

from typing import List, Dict, Any

# Try to import ddgs, fallback to duckduckgo_search
try:
    from ddgs import DDGS as DDS
except ImportError:
    try:
        from duckduckgo_search import DDGS as DDS
    except ImportError:
        DDS = None


class WebSearch:
    """Web search for latest LLM information"""

    def __init__(self):
        self.ddgs = DDS() if DDS else None

    def search_llms(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web for LLM information

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with title, url, and snippet
        """
        if not self.ddgs:
            return []

        try:
            results = list(self.ddgs.text(query, max_results=max_results))
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
        search_query = f"best LLMs for {workflow} agentic AI workflow tool calling 2024 2025"
        results = self.search_llms(search_query, max_results=10)

        return {
            "query": workflow,
            "results": results,
            "summary": self._generate_summary(results)
        }

    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary from search results"""
        if not results:
            return "No search results found."

        summary_parts = []
        for i, result in enumerate(results[:5], 1):
            title = result.get("title", "")
            snippet = result.get("body", "")[:100] if result.get("body") else ""
            summary_parts.append(f"{i}. {title}: {snippet}...")

        return "\n".join(summary_parts)


def search_web(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Convenience function for web search"""
    searcher = WebSearch()
    return searcher.search_llms(query, max_results)


def get_llm_recommendations_search(workflow: str) -> Dict[str, Any]:
    """Get web search results for LLM recommendations"""
    searcher = WebSearch()
    return searcher.search_llm_recommendations(workflow)


# Test if web search is available
def is_web_search_available() -> bool:
    """Check if web search is working"""
    try:
        searcher = WebSearch()
        results = searcher.search_llms("test", max_results=1)
        return len(results) > 0
    except:
        return False
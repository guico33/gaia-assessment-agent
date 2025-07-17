"""
Web Search Tool using Tavily API for research and fact-finding.
Provides comprehensive web search capabilities for GAIA questions.
"""

import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from tavily import TavilyClient

from ..config import Config


class WebSearchTool:
    """
    Handles web search operations using Tavily API.
    """

    def __init__(self):
        """Initialize the web search tool with Tavily API."""
        self.tavily_api_key = Config.get_tavily_api_key()
        if self.tavily_api_key:
            self.client = TavilyClient(api_key=self.tavily_api_key)
        else:
            self.client = None
            print("Warning: TAVILY_API_KEY not found. Web search will be limited.")

    def search(
        self,
        query: str,
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> str:
        """
        Perform a web search using Tavily API.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search

        Returns:
            String containing formatted search results
        """
        if not self.client:
            return "Error: Web search unavailable (TAVILY_API_KEY not configured)"

        try:
            # Prepare search parameters
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
            }

            # Add domain filters if specified
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            # Perform the search
            response = self.client.search(**search_params)

            # Format the results
            return self._format_search_results(response, query)

        except Exception as e:
            return f"Error performing web search: {str(e)}"

    def search_specific_site(
        self, query: str, domain: str, max_results: int = 3
    ) -> str:
        """
        Search within a specific website domain.

        Args:
            query: Search query
            domain: Domain to search within (e.g., "wikipedia.org")
            max_results: Maximum number of results

        Returns:
            String containing formatted search results
        """
        return self.search(query, max_results, include_domains=[domain])

    def search_wikipedia(self, query: str, max_results: int = 3) -> str:
        """
        Search specifically on Wikipedia.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            String containing Wikipedia search results
        """
        return self.search_specific_site(query, "wikipedia.org", max_results)

    def search_academic(self, query: str, max_results: int = 3) -> str:
        """
        Search for academic and research content.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            String containing academic search results
        """
        academic_domains = [
            "scholar.google.com",
            "arxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "jstor.org",
            "academia.edu",
        ]
        return self.search(query, max_results, include_domains=academic_domains)

    def search_news(self, query: str, max_results: int = 3) -> str:
        """
        Search for news and current events.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            String containing news search results
        """
        # Let Tavily handle news search with recency
        return self.search(f"news {query}", max_results)

    def _format_search_results(self, response: Dict[str, Any], query: str) -> str:
        """
        Format search results into a readable string.

        Args:
            response: Tavily API response
            query: Original search query

        Returns:
            Formatted string with search results
        """
        try:
            formatted_results = f"Web Search Results for: {query}\n\n"

            # Add answer if available
            if "answer" in response and response["answer"]:
                formatted_results += f"Quick Answer: {response['answer']}\n\n"

            # Add search results
            if "results" in response:
                for i, result in enumerate(response["results"], 1):
                    formatted_results += f"{i}. {result.get('title', 'No title')}\n"
                    formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
                    formatted_results += f"   Content: {result.get('content', 'No content')[:300]}...\n\n"

            return formatted_results

        except Exception as e:
            return f"Error formatting search results: {str(e)}"

    def get_search_suggestions(self, query: str) -> List[str]:
        """
        Get search suggestions for a given query.

        Args:
            query: Base query to get suggestions for

        Returns:
            List of suggested search queries
        """
        # Generate some common search variations
        suggestions = [
            f"{query} definition",
            f"{query} examples",
            f"{query} latest news",
            f"{query} wikipedia",
            f"what is {query}",
            f"how to {query}",
            f"{query} facts",
        ]

        return suggestions[:5]  # Return top 5 suggestions


# Global instance
_web_search = WebSearchTool()


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search to find information.

    Args:
        query: Search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        String containing formatted search results
    """
    return _web_search.search(query, max_results)


@tool
def search_wikipedia(query: str) -> str:
    """
    Search specifically on Wikipedia for factual information.

    Args:
        query: Search query

    Returns:
        String containing Wikipedia search results
    """
    return _web_search.search_wikipedia(query)


@tool
def search_specific_site(query: str, domain: str) -> str:
    """
    Search within a specific website domain.

    Args:
        query: Search query
        domain: Domain to search within (e.g., "example.com")

    Returns:
        String containing search results from the specified domain
    """
    return _web_search.search_specific_site(query, domain)


@tool
def search_academic(query: str) -> str:
    """
    Search for academic and research content.

    Args:
        query: Search query

    Returns:
        String containing academic search results
    """
    return _web_search.search_academic(query)


@tool
def search_news(query: str) -> str:
    """
    Search for news and current events.

    Args:
        query: Search query

    Returns:
        String containing news search results
    """
    return _web_search.search_news(query)


# Export the tools for use in the main tools module
__all__ = [
    "web_search",
    "search_wikipedia",
    "search_specific_site",
    "search_academic",
    "search_news",
    "WebSearchTool",
]

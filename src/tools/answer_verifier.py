"""
Answer Verifier Tool for GAIA format compliance and answer refinement.
Ensures all answers follow GAIA specifications and formatting requirements.
"""

from typing import Any, Dict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from ..llm_provider import get_llm


class AnswerVerifier:
    """
    Handles answer verification and formatting for GAIA compliance.
    """

    def __init__(self):
        """Initialize the answer verifier with LLM."""
        self.llm = get_llm()

    def verify_and_format_answer(
        self, question: str, current_answer: str, context: str = ""
    ) -> str:
        """
        Verify and format an answer according to GAIA specifications.

        Args:
            question: The original question
            current_answer: Current answer to verify and format
            context: Additional context from the analysis

        Returns:
            String containing the verified and formatted answer
        """
        try:
            # Create GAIA formatting prompt
            gaia_prompt = f"""
            You are a GAIA answer formatter. Your task is to take an answer and format it according to GAIA specifications.

            GAIA FORMATTING RULES:
            1. NUMBERS: No commas (write 1000 not 1,000), no units unless specifically requested
            2. STRINGS: No articles (write "Einstein" not "the Einstein"), NO ABBREVIATIONS (expand all abbreviations to full form), write digits in plain text unless specified
            3. LISTS: Comma separated, apply above rules to each element
            4. RETURN FORMAT: Provide only the raw answer without any prefix or template
            5. Be as concise as possible - typically just a number, name, or short phrase

            CRITICAL: NO ABBREVIATIONS RULE - Always expand abbreviations:
            - "Bvd" → "Boulevard" (e.g., "Bvd. de la Liberté" → "Boulevard de la Liberté")
            - "Dr." → "Doctor" 
            - "NYC" → "New York City"
            - "USA" → "United States"
            - "UK" → "United Kingdom"
            - "Mt." → "Mount"

            Examples:
            - For "How many albums?" → "5"
            - For "What city?" → "New York" 
            - For "Where is the museum?" with answer "St. Petersburg" → "Saint Petersburg"
            - For "List the names" → "Smith, Johnson, Williams"
            - For "What percentage?" → "25" (no % unless requested)

            Question: {question}
            
            Current Answer: {current_answer}
            
            Context: {context}
            
            IMPORTANT: If the question asks for "without abbreviations" or similar, you MUST expand all abbreviated forms. 
            
            Please format this answer according to GAIA specifications. Provide only the raw answer without any "FINAL ANSWER:" prefix.
            """

            # Get LLM response
            messages = [SystemMessage(content=gaia_prompt)]
            response = self.llm.invoke(messages)

            # Extract content from response
            if hasattr(response, "content"):
                formatted_answer = str(response.content).strip()
            else:
                formatted_answer = str(response).strip()

            # Remove any "FINAL ANSWER:" prefix if it was added by mistake
            if "FINAL ANSWER:" in formatted_answer:
                formatted_answer = formatted_answer.replace("FINAL ANSWER:", "").strip()

            return formatted_answer

        except Exception:
            # Fallback formatting if LLM fails
            return self._fallback_format_answer(current_answer)

    def _fallback_format_answer(self, answer: str) -> str:
        """Minimal fallback formatting when LLM processing fails."""
        # Just remove any "FINAL ANSWER:" prefix and basic cleanup
        answer = answer.strip()

        # Remove any "FINAL ANSWER:" prefix if present
        if "FINAL ANSWER:" in answer:
            answer = answer.replace("FINAL ANSWER:", "").strip()

        return answer

    def assess_answer_quality(
        self, question: str, answer: str, context: str = ""
    ) -> Dict[str, Any]:
        """
        Assess the quality of an answer and determine if more processing is needed.

        Args:
            question: The original question
            answer: Current answer to assess
            context: Additional context from the analysis

        Returns:
            Dictionary with quality assessment
        """
        try:
            quality_prompt = f"""
            You are a GAIA answer quality assessor. Evaluate whether this answer is satisfactory or needs more processing.

            Question: {question}
            
            Current Answer: {answer}
            
            Context: {context}
            
            Assess the answer quality based on:
            1. Completeness - Does it fully answer the question?
            2. Accuracy - Is the answer correct based on available information?
            3. Specificity - Is it specific enough for GAIA evaluation?
            4. Format - Does it follow GAIA formatting requirements?
            
            Respond with either:
            - "SATISFACTORY" if the answer is ready for submission
            - "NEEDS_MORE_PROCESSING" if more analysis is needed
            
            Also provide a brief reason for your assessment.
            """

            messages = [SystemMessage(content=quality_prompt)]
            response = self.llm.invoke(messages)

            # Extract content from response
            if hasattr(response, "content"):
                assessment = str(response.content).strip()
            else:
                assessment = str(response).strip()

            # Parse the assessment
            if "SATISFACTORY" in assessment.upper():
                return {
                    "status": "satisfactory",
                    "assessment": assessment,
                    "needs_more_processing": False,
                }
            else:
                return {
                    "status": "needs_more_processing",
                    "assessment": assessment,
                    "needs_more_processing": True,
                }

        except Exception as e:
            # Fallback assessment
            return {
                "status": "needs_more_processing",
                "assessment": f"Error in assessment: {str(e)}",
                "needs_more_processing": True,
            }

    def suggest_improvements(
        self, question: str, answer: str, context: str = ""
    ) -> str:
        """
        Suggest improvements for an answer that needs more processing.

        Args:
            question: The original question
            answer: Current answer
            context: Additional context

        Returns:
            String with improvement suggestions
        """
        try:
            improvement_prompt = f"""
            You are a GAIA answer improvement advisor. Suggest specific improvements for this answer.

            Question: {question}
            
            Current Answer: {answer}
            
            Context: {context}
            
            Suggest specific improvements such as:
            - What additional information is needed?
            - What tools should be used for more analysis?
            - What aspects of the question are not fully addressed?
            - How can the answer be more specific or accurate?
            
            Provide concrete, actionable suggestions.
            """

            messages = [SystemMessage(content=improvement_prompt)]
            response = self.llm.invoke(messages)

            # Extract content from response
            if hasattr(response, "content"):
                suggestions = str(response.content).strip()
            else:
                suggestions = str(response).strip()

            return suggestions

        except Exception as e:
            return f"Error generating improvement suggestions: {str(e)}"


# Global instance
_answer_verifier = AnswerVerifier()


@tool
def verify_gaia_answer(question: str, current_answer: str, context: str = "") -> str:
    """
    Verify and format an answer according to GAIA specifications.

    Args:
        question: The original question
        current_answer: Current answer to verify and format
        context: Additional context from the analysis

    Returns:
        String containing the verified and formatted answer
    """
    return _answer_verifier.verify_and_format_answer(question, current_answer, context)


@tool
def assess_answer_quality(question: str, answer: str, context: str = "") -> str:
    """
    Assess the quality of an answer and determine if more processing is needed.

    Args:
        question: The original question
        answer: Current answer to assess
        context: Additional context from the analysis

    Returns:
        String containing quality assessment
    """
    assessment = _answer_verifier.assess_answer_quality(question, answer, context)
    return f"Quality Status: {assessment['status']}\nAssessment: {assessment['assessment']}"


@tool
def suggest_answer_improvements(question: str, answer: str, context: str = "") -> str:
    """
    Suggest improvements for an answer that needs more processing.

    Args:
        question: The original question
        answer: Current answer
        context: Additional context

    Returns:
        String with improvement suggestions
    """
    return _answer_verifier.suggest_improvements(question, answer, context)


# Export the tools for use in the main tools module
__all__ = [
    "verify_gaia_answer",
    "assess_answer_quality",
    "suggest_answer_improvements",
    "AnswerVerifier",
]

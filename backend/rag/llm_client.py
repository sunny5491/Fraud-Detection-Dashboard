"""
LLM Client for generating grounded fraud investigation answers using Groq API and Llama 3.3 70B.
"""
import os
import sys
from typing import List

from groq import Groq

# Ensure project root config is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    import config
except ImportError:
    sys.path.append(os.getcwd())
    import config

SYSTEM_PROMPT = (
    "You are a professional fraud-investigation assistant. "
    "Answer the investigator's question ONLY using the provided context about the user case. "
    "If the provided context does not cover or contain enough information to answer the question, "
    "state 'I don't have enough information to answer that.' "
    "Never invent, hallucinate, or assume any facts or numbers not explicitly present in the context."
)


def generate_answer(user_id: str, question: str, context_chunks: List[str]) -> str:
    """
    Generate a grounded natural-language answer to an investigator's question based on retrieved context chunks.
    
    Args:
        user_id: Target user ID string.
        question: Investigator's question string.
        context_chunks: List of context chunk strings retrieved from vector store.
        
    Returns:
        Generated answer string from Groq Llama 3.3 70B, or fallback message on failure.
    """
    if not user_id or not question:
        return "I don't have enough information to answer that."

    api_key = getattr(config, "GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_key_here":
        return "The explanation assistant is temporarily unavailable — please check GROQ_API_KEY is set."

    formatted_context = "\n\n".join(context_chunks) if context_chunks else "No context provided."
    user_prompt = (
        f"Context for User Case ({user_id}):\n"
        f"-------------------------------\n"
        f"{formatted_context}\n"
        f"-------------------------------\n\n"
        f"Investigator Question: {question}"
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        if response and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        return "The explanation assistant is temporarily unavailable — please check GROQ_API_KEY is set."
    except Exception:
        return "The explanation assistant is temporarily unavailable — please check GROQ_API_KEY is set."


if __name__ == "__main__":
    fake_user = "USER00000001"
    fake_chunks = [
        "## Profile Overview for User USER00000001\n- Risk Score: 88.5 / 100 (High Risk)\n- Total Financial Exposure: $1250.00\n- Total Returns Count: 12",
        "## SHAP Risk Analysis\n- Feature 'High-Value Item Ratio': SHAP contribution 0.45 (increases risk)\n- Feature 'Return Frequency': SHAP contribution 0.35 (increases risk)"
    ]
    fake_question = "Why is this user marked as high risk and what is their total financial exposure?"

    print(f"Testing LLM Client for user: {fake_user}")
    print(f"Question: {fake_question}\n")
    answer = generate_answer(fake_user, fake_question, fake_chunks)
    print("Response:")
    print(answer)

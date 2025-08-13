import json
import time
from typing import Dict, Any, List
import re

def call_model(messages: List[Dict[str, str]], model_name: str = "gpt-oss:20b", temperature: float = 1.0) -> Dict[str, Any]:
    try:
        import ollama  # local inference via Ollama daemon
    except Exception as e:
        return {
            "content": f"Ollama client not available: {e}. "
                       f"Install with 'pip install ollama' and ensure the Ollama app/daemon is running.",
            "response_time": 0.0,
            "success": False,
        }

    try:
        start_time = time.time()
        options = {
            "temperature": float(temperature),
            "top_p": 1.0,
        }
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=options
        )
        end_time = time.time()

        # Normalize response content
        if isinstance(response, dict) and "message" in response:
            content = response["message"].get("content", "")
        else:
            # Fallback for any different client shape
            content = str(response)

        return {
            "content": content,
            "response_time": end_time - start_time,
            "success": True,
        }
    except Exception as e:
        return {
            "content": f"Error during model call: {e}",
            "response_time": 0.0,
            "success": False,
        }


def parse_reasoning_response(content: str) -> Dict[str, str]:
    """
    Attempts to separate any 'reasoning' from the final 'answer' in model output.
    If no explicit reasoning markers are present, returns the original content as the answer.
    """
    patterns = [
        r"<thinking>(.*?)</thinking>",
        r"Reasoning:(.*?)(?=\n\n|\nAnswer:|\nConclusion:|\Z)",
        r"Let me think.*?:(.*?)(?=\n\n|\nFinal|Answer:|\Z)",
    ]

    reasoning = ""
    answer = content

    for pat in patterns:
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip()
            # Remove full match from content to leave the 'answer'
            answer = content.replace(m.group(0), "").strip()
            break

    # Heuristic fallback split if long multi-line text
    if not reasoning and "\n" in content and len(content.splitlines()) > 3:
        lines = content.splitlines()
        for i, l in enumerate(lines):
            ll = l.lower()
            if any(k in ll for k in ["therefore", "in conclusion", "final answer", "answer:"]):
                reasoning = "\n".join(lines[:i]).strip()
                answer = "\n".join(lines[i:]).strip()
                break

    return {
        "reasoning": reasoning if reasoning else "No explicit reasoning detected.",
        "answer": answer if answer else content,
    }


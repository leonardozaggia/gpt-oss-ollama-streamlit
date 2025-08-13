import time
from typing import Dict, Any, List
import re
def call_model(messages: List[Dict[str, str]], model_name: str, temperature: float = 1.0, effort: str = "medium") -> Dict[str, Any]:
    """Call a local model via Ollama's Python client. Returns normalized dict."""
    try:
        import ollama
    except Exception as e:
        return {
            "content": f"Ollama client not available: {e}. "
                       f"Install with 'pip install ollama' and ensure the Ollama app/daemon is running.",
            "response_time": 0.0,
            "success": False,
        }

    # Map effort → sampling style (model-agnostic)
    effort = (effort or "medium").lower()
    if effort == "low":
        temp, top_p = max(0.1, temperature * 0.6), 0.9
    elif effort == "high":
        temp, top_p = min(2.0, temperature * 1.2), 1.0
    else:
        temp, top_p = temperature, 0.95

    try:
        import time
        start_time = time.time()
        options = {"temperature": float(temp), "top_p": float(top_p)}
        response = ollama.chat(model=model_name, messages=messages, options=options)
        dt = time.time() - start_time
        content = response["message"].get("content", "") if isinstance(response, dict) and "message" in response else str(response)
        return {"content": content, 
                "response_time": dt, 
                "success": True}
    except Exception as e:
        return {"content": f"Error during model call: {e}", "response_time": 0.0, "success": False}


def parse_reasoning_response(content: str) -> Dict[str, str]:
    """
    Attempts to separate 'reasoning' from 'answer' when the model emits chain-of-thought.
    If nothing is detected, returns the original content as the answer.
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
            answer = content.replace(m.group(0), "").strip()
            break

    # Heuristic fallback
    if not reasoning and "\n" in content and len(content.splitlines()) > 3:
        lines = content.splitlines()
        for i, l in enumerate(lines):
            ll = l.lower()
            if any(k in ll for k in ["therefore", "in conclusion", "final answer", "answer:"]):
                reasoning = "\n".join(lines[:i]).strip()
                answer = "\n".join(lines[i:]).strip()
                break

    return {"reasoning": reasoning if reasoning else "", "answer": answer if answer else content}


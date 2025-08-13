import streamlit as st
from typing import Dict, List
from utility import call_model, parse_reasoning_response

def main():
    st.set_page_config(page_title="GPT-OSS • Local Chat (Ollama)", layout="wide", page_icon="💬")

    # Styles
    st.markdown(
        """
    <style>
    :root { --muted: rgba(0,0,0,0.04); }
    .reasoning-box { background: rgba(0, 212, 170, 0.06); padding: 1rem; border-radius: 12px; border-left: 4px solid #00d4aa; }
    .answer-box { background: rgba(0, 123, 255, 0.06); padding: 1rem; border-radius: 12px; border-left: 4px solid #007bff; }
    .metric-card { background: var(--muted); padding: 0.75rem 1rem; border-radius: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .chat-message { background: var(--muted); padding: 0.6rem 0.8rem; border-radius: 10px; margin-bottom: 0.35rem; }
    .user-message { border-left: 4px solid #2196f3; }
    .assistant-message { border-left: 4px solid #9c27b0; }
    .badge { display:inline-block; padding: .15rem .5rem; border-radius: 999px; background: var(--muted); margin-right:.25rem; font-size:.85em;}
    </style>
    """,
        unsafe_allow_html=True
    )

    # State
    if "history" not in st.session_state:
        st.session_state.history = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-oss:20b"


    with st.sidebar:
        st.header("Configuration")
        st.caption("Choose any Ollama model. Works with OSS models that **don't emit reasoning** as well.")

        preset = st.selectbox(
            "Preset model",
            ["gpt-oss:20b", "gpt-oss:120b", "llama3.1:8b", "phi3:latest", "qwen2.5:7b", "mistral:7b", "custom", "codellama:7b", "gemma:2b"],
            index=0,
            help="Pick a known model or choose 'custom' and type your own below."
        )
        custom = st.text_input("Custom model name (overrides preset if non-empty)", value="")
        model_name = custom.strip() if custom.strip() else preset
        st.session_state.model_name = model_name

        # Reasoning handling
        reasoning_mode = st.radio(
            "Reasoning display",
            ["Auto-detect", "Always hide"],
            index=0,
            help="If your model doesn't output chain-of-thought, choose 'Always hide' to skip parsing."
        )

        effort = st.selectbox("Reasoning Effort", ["low", "medium", "high"], index=1,
                      help="Low = concise, Medium = brief rationale, High = detailed reasoning (if model supports it)")

        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)

        st.markdown("---")
        if st.button("Clear Conversation"):
            st.session_state.history = []
            st.rerun()

    # Header
    colA, colB = st.columns([3, 2])
    with colA:
        st.title("💬 Local Chat (Ollama)")
        st.caption("Flexible model selection • Optional reasoning display • Lightweight metrics")
    with colB:
        st.markdown(
            f"""
    <div class="metric-card">
    <div class="badge">Model</div> <strong>{st.session_state.model_name}</strong><br>
    <div class="badge">Temp</div> {temperature} &nbsp; <div class="badge">Mode</div> {reasoning_mode}
    </div>
    """,
            unsafe_allow_html=True
        )

    # Examples / Prompt
    examples = [
        "",
        "If a train travels 120 km in 1.5 hours, then 80 km in 45 minutes, what's its average speed?",
        "Prove that √2 is irrational.",
        "Write a function to find the longest palindromic substring.",
        "Explain quantum entanglement in simple terms.",
        "How would you design a recommendation system?",
    ]

    selected_from_dropdown = st.selectbox("Examples:", examples)
    prefill = selected_from_dropdown if selected_from_dropdown else ""

    question = st.text_area(
        "Ask anything:",
        value=prefill,
        height=100,
        placeholder="Type a question and click 'Send'. You can switch models freely (e.g., llama3.1:8b, mistral:7b, phi3...).",
    )

    send = st.button("Send", type="primary")

    if send and question.strip():
        if effort == "low":
            system_prompt = "You are a helpful assistant. Provide a concise answer. Avoid showing your reasoning unless strictly necessary."
        elif effort == "high":
            system_prompt = "You are a helpful assistant. Think carefully and explain your steps briefly before your final answer."
        else:
            system_prompt = "You are a helpful assistant. Provide a short rationale then a clear final answer."

        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        msgs.extend(st.session_state.history[-6:])  # maintain context
        msgs.append({"role": "user", "content": question})

        with st.spinner(f"Thinking with {st.session_state.model_name}..."):
            res = call_model(msgs, model_name=st.session_state.model_name, temperature=temperature, effort=effort)

        if res["success"]:
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": res["content"]})

            # Metrics
            col1, col2 = st.columns([3, 1])
            with col1:
                if reasoning_mode == "Auto-detect":
                    parsed = parse_reasoning_response(res["content"])
                    # Show reasoning only if something meaningful detected
                    if parsed["reasoning"] and parsed["reasoning"].strip():
                        st.markdown("### Reasoning (detected)")
                        st.markdown(f"<div class='reasoning-box'>{parsed['reasoning']}</div>", unsafe_allow_html=True)
                    st.markdown("### Answer")
                    st.markdown(f"<div class='answer-box'>{parsed['answer']}</div>", unsafe_allow_html=True)
                else:
                    # Always hide reasoning
                    st.markdown("### Answer")
                    st.markdown(f"<div class='answer-box'>{res['content']}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("#### Metrics")
                st.markdown(
                    f"""
    <div class="metric-card">
    <strong>Time:</strong> {res['response_time']:.2f}s<br>
    <strong>Model:</strong> {st.session_state.model_name}<br>
    <strong>Temp:</strong> {temperature}
    </div>
    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error(res["content"])

    # Conversation history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Conversation")
        for msg in st.session_state.history[-8:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {content}</div>", unsafe_allow_html=True)
            elif role == "assistant":
                st.markdown(f"<div class='chat-message assistant-message'><strong>Assistant:</strong> {content}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='text-align:center'><strong>Powered locally via Ollama</strong></div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
import streamlit as st
from typing import Dict, List
from utility import call_model, parse_reasoning_response

def main():
    # ---------------- Streamlit App ----------------

    st.set_page_config(page_title="GPT-OSS Chat Demo", layout="wide", page_icon="💬")

    # Lightweight styling
    st.markdown(
        """
    <style>
    .reasoning-box {
        background: rgba(0, 212, 170, 0.06);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00d4aa;
    }
    .answer-box {
        background: rgba(0, 123, 255, 0.06);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    .metric-card {
        background: rgba(0,0,0,0.03);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: left;
    }
    .chat-message {
        background: rgba(0,0,0,0.03);
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.35rem;
    }
    .user-message {
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        border-left: 4px solid #9c27b0;
    }
    </style>
    """,
        unsafe_allow_html=True
    )

    # Initialize history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar — configuration
    with st.sidebar:
        st.header("Configuration")

        model_choice = st.selectbox(
            "Model",
            ["gpt-oss:20b", "gpt-oss:120b"],
            help="Choose between 20B (faster) or 120B (more capable)",
        )

        effort = st.selectbox(
            "Reasoning Effort",
            ["low", "medium", "high"],
            index=1,
            help="Controls how much reasoning the model attempts to show",
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            1.0,
            0.1,
            help="Higher is more random; lower is more deterministic",
        )

        show_reasoning = st.checkbox(
            "Show Reasoning (if present)",
            True,
            help="Display any explicit reasoning the model outputs",
        )

        show_metrics = st.checkbox(
            "Show Performance Metrics",
            True,
            help="Display response time and model info",
        )

        st.markdown("---")
        if st.button("Clear Conversation"):
            st.session_state.history = []
            st.rerun()

    # Main title and examples
    st.title("💬 GPT-OSS Interactive Chat")
    examples = [
        "",
        "If a train travels 120 km in 1.5 hours, then 80 km in 45 minutes, what's its average speed?",
        "Prove that √2 is irrational.",
        "Write a function to find the longest palindromic substring.",
        "Explain quantum entanglement in simple terms.",
        "How would you design a recommendation system?",
    ]

    selected_from_dropdown = st.selectbox("Choose from examples:", examples)

    # Prefill input if a preset is chosen
    question_value = ""
    if hasattr(st.session_state, "selected_example"):
        question_value = st.session_state.selected_example
        del st.session_state.selected_example
    elif selected_from_dropdown:
        question_value = selected_from_dropdown

    question = st.text_area(
        "Or enter your question:",
        value=question_value,
        height=100,
        placeholder="Ask anything! Try different reasoning effort levels to see how the model's thinking changes...",
    )

    # Submit
    submit_button = st.button("Ask GPT-OSS", type="primary")

    if submit_button and question.strip():
        system_prompts = {
            "low": "You are a helpful assistant. Provide concise, direct answers.",
            "medium": f"You are a helpful assistant. Show brief reasoning before your answer. Reasoning effort: {effort}.",
            "high": f"You are a helpful assistant. Think through the problem step by step. Provide your final answer clearly. Reasoning effort: {effort}.",
        }

        # Build message list (keep last few turns for context)
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompts[effort]}]
        msgs.extend(st.session_state.history[-6:])
        msgs.append({"role": "user", "content": question})

        with st.spinner(f"GPT-OSS thinking ({effort} effort)..."):
            res = call_model(msgs, model_name=model_choice, temperature=temperature)

        if res["success"]:
            parsed = parse_reasoning_response(res["content"])
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": res["content"]})

            if show_metrics:
                st.markdown("#### Metrics")
                st.markdown(
                    f"""
    <div class="metric-card">
    <strong>Time:</strong> {res['response_time']:.2f}s<br>
    <strong>Model:</strong> {model_choice}<br>
    <strong>Effort:</strong> {effort.title()}
    </div>
    """,
                    unsafe_allow_html=True,
                )

            if show_reasoning and parsed["reasoning"] != "No explicit reasoning detected.":
                st.markdown("### Reasoning (as provided by the model)")
                st.markdown(f"<div class='reasoning-box'>{parsed['reasoning']}</div>", unsafe_allow_html=True)

            st.markdown("### Answer")
            st.markdown(f"<div class='answer-box'>{parsed['answer']}</div>", unsafe_allow_html=True)

        else:
            st.error(res["content"])

    # Conversation history (last few turns)
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Conversation History")
        recent_history = st.session_state.history[-8:]  # last 4 exchanges
        for i, msg in enumerate(recent_history):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                st.markdown(
                    f"<div class='chat-message user-message'><strong>You:</strong> {content}</div>",
                    unsafe_allow_html=True,
                )
            elif role == "assistant":
                parsed_hist = parse_reasoning_response(content)
                with st.expander("Assistant Response", expanded=False):
                    if parsed_hist["reasoning"] != "No explicit reasoning detected.":
                        st.markdown("**Reasoning (if present):**")
                        st.code(parsed_hist["reasoning"], language="text")
                    st.markdown("**Final Answer:**")
                    st.markdown(parsed_hist["answer"])

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center'><strong>Powered locally via Ollama • GPT-OSS</strong></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
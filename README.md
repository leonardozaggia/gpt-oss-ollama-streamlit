"""
README — How to Set Up and Run GPT-OSS Locally with Ollama
==========================================================

This README explains the **full setup process** you must follow *before*
cloning the repository and running the GPT-OSS Streamlit demo locally.
It assumes no prior local Ollama configuration.

----------------------------------------------------------------------
1. Overview
----------------------------------------------------------------------
GPT-OSS is OpenAI’s open-weight model series (Apache 2.0 licensed) designed
for powerful reasoning, transparency, and flexible local deployment.
We will use:
    - Ollama: to handle local model download, quantization, and execution.
    - Streamlit: to provide an interactive web UI for the chatbot demo.

Two GPT-OSS model variants exist:
    • gpt-oss:20b  → Fast, fits on consumer GPUs or CPUs with 16GB+ RAM
    • gpt-oss:120b → More capable, needs 80GB+ GPU memory

This guide prepares your machine to run **gpt-oss:20b** by default.

----------------------------------------------------------------------
2. Prerequisites
----------------------------------------------------------------------
Before you start, ensure you have:
    - macOS (Apple Silicon or Intel), Windows, or Linux machine
    - Python 3.9+ installed
    - 16GB+ system memory for gpt-oss:20b (80GB+ GPU VRAM for gpt-oss:120b)
    - Git installed
    - An internet connection (first-time model download)

----------------------------------------------------------------------
3. Step-by-Step Setup BEFORE Cloning the Repo
----------------------------------------------------------------------

STEP 1 — Install Ollama
-----------------------
1. Visit: https://ollama.com/download
2. Download and install Ollama for your OS.
3. After installation, open a terminal and verify:
       ollama --version
   You should see the installed version number.

STEP 2 — Pull the GPT-OSS Model
-------------------------------
For the 20B model (recommended for most users):
    ollama pull gpt-oss:20b

For the 120B model (requires 80GB GPU VRAM):
    ollama pull gpt-oss:120b

**Tip:** You can pull both models, but only one is needed to run the demo.

STEP 3 — Verify Model Works
---------------------------
Run a quick interactive test:
    ollama run gpt-oss:20b
At the prompt, try:
    Prove that √2 is irrational.

If you see a valid answer, your model setup is correct.

----------------------------------------------------------------------
4. Clone the Repository
----------------------------------------------------------------------
Once Ollama and GPT-OSS are ready, clone your code repository:
    git clone <your_repo_url>
    cd <your_repo_folder>

----------------------------------------------------------------------
5. Python Environment Setup
----------------------------------------------------------------------
It’s recommended to use a virtual environment:

    python -m venv venv
    source venv/bin/activate        # macOS/Linux
    venv\Scripts\activate           # Windows

Then install dependencies:
    pip install -r requirements.txt

If there is no requirements.txt, install manually:
    pip install streamlit ollama

----------------------------------------------------------------------
6. Running the Demo
----------------------------------------------------------------------
Run the Streamlit app:
    streamlit run gpt_oss_demo.py

This will:
    - Open a local browser window (http://localhost:8501 by default)
    - Allow you to choose model variant, reasoning effort, temperature
    - Show or hide chain-of-thought reasoning
    - Interact with GPT-OSS fully offline after first model pull

----------------------------------------------------------------------
7. Troubleshooting
----------------------------------------------------------------------
• If `ollama` is not found:
    - Ensure Ollama is installed and in your PATH.
• If Streamlit says a package is missing:
    - Install it with pip and restart: pip install <package_name>
• If model download is slow:
    - It may be due to network speed; model files are large (GBs).

----------------------------------------------------------------------
8. Conclusion
----------------------------------------------------------------------
After completing these steps, you can run GPT-OSS locally with full control,
privacy, and transparency. You can easily extend the app by modifying
`gpt_oss_demo.py` for more features such as:
    - Additional reasoning modes
    - API endpoints
    - Logging and analytics

Enjoy building with GPT-OSS!
"""

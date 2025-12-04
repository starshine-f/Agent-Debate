# Agent Debate（多智能体辩论赛）
Multi-Agent Debate Demo (FastAPI + LangGraph + LangChain OpenAI)

> A hands-on demo for multi-agent *debate-style reasoning*, easy to run and easy to extend.

![智能体辩论效果图 1](img/AI-Debate.png)
![智能体辩论效果图 2](img/AI-Debate2.png)

---

## ✨ Features

### 🧠 AI vs AI Multi-Agent Debate

- A judge plus 4 debaters for the Pro side and 4 for the Con side (9 agents in total).
- Each position (1st–4th speaker) has a different role and persona, fully configurable.
- Debate flow is orchestrated via a directed graph / FSM built with **LangGraph**.
- Supports **NDJSON streaming** for real-time UI updates.

---

### 👤 Human vs AI Debate

- You can freely choose to argue for the **Pro** or **Con** side.
- The AI side cycles through 4 debaters: opening, rebuttal, examples, and summary.
- Chat-style interaction with multi-turn debate rounds.

---

### 🖥 Built-in Frontend

- Comes with a minimal HTML page (`index.html`), no separate frontend project needed.
- Uses `fetch` + NDJSON streaming to update debate turns progressively.
- Default static assets (avatars, etc.) live under `app/static/`.

---

### 🧩 Tech Stack

- **Backend**
  - FastAPI
  - LangGraph
  - LangChain / LangChain OpenAI
- **Model Providers**
  - OpenAI / DeepSeek / DashScope (Qwen) / Kimi / BigModel (GLM) and others
- **Frontend**
  - HTML + JavaScript

---

## 📁 Project Layout

```text
.
├─ requirements.txt
├─ README.md
├─ .env                           # Stores model API keys & base URLs
└─ app/
   ├─ __init__.py
   ├─ api.py                      # FastAPI entrypoint
   ├─ agent.py                    # DebateState / AgentRole / speak_with_role
   ├─ graph.py                    # LangGraph FSM for debate flow
   ├─ config.py                   # Model profiles & personas
   ├─ demo.py                     # CLI demo
   ├─ index.html                  # Frontend page
   └─ static/                     # Static assets (avatars, etc.)
```

---

## 🧱 Environment

- Python **3.10+** is recommended (ideally inside a virtual environment).
- Requires network access to the model providers you want to use (OpenAI / DeepSeek / DashScope / Kimi / BigModel, etc.).

---

## 📦 Installing Dependencies

### 1️⃣ Create a virtualenv 

```bash
python -m venv .venv
# Windows
# .venv\Scriptsctivate
# macOS / Linux
# source .venv/bin/activate
```

### 2️⃣ Install requirements

```bash
pip install -r requirements.txt
```

---

## 🔧 Configure Environment Variables

All model settings live in `app/config.py` under `MODEL_PROFILES`:

```python
MODEL_PROFILES = {
    "gpt4.1": {
        "model": "gpt-4.1",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "label": "OpenAI GPT-4.1",
        "group": "OpenAI",
    },
    "deepseek-chat": {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "label": "DeepSeek Chat",
        "group": "DeepSeek",
    },
    "deepseek-reasoner": {
        "model": "deepseek-reasoner",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "label": "DeepSeek Reasoner",
        "group": "DeepSeek",
    },
    "qwen3-max": {
        "model": "qwen3-max",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "label": "Qwen3-Max",
        "group": "DashScope",
    },
    "kimi-k2-turbo-preview": {
        "model": "kimi-k2-turbo-preview",
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url_env": "MOONSHOT_BASE_URL",
        "label": "Kimi K2 Turbo Preview",
        "group": "Kimi",
    },
    "glm-4.5": {
        "model": "glm-4.5",
        "api_key_env": "BIGMODEL_API_KEY",
        "base_url_env": "BIGMODEL_BASE_URL",
        "label": "GLM-4.5",
        "group": "BigModel",
    },
}
```

You can provide all keys & base URLs via environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `MOONSHOT_API_KEY`
- `MOONSHOT_BASE_URL`
- `BIGMODEL_API_KEY`
- `BIGMODEL_BASE_URL`

---

### 📄 Use a `.env` File (Recommended)

Create a `.env` file at the project root, for example:

```env
# OpenAI
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.openai.com/v1

# DeepSeek
DEEPSEEK_API_KEY=ds-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# DashScope (Qwen)
DASHSCOPE_API_KEY=ds-xxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Kimi Moonshot
MOONSHOT_API_KEY=ms-xxxx
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1

# BigModel (GLM)
BIGMODEL_API_KEY=glm-xxxx
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

`config.py` already calls `load_dotenv()`, so values from `.env` are loaded automatically.

---

## 🖥 Command-Line Demo

Try an AI vs AI debate quickly in your terminal:

```bash
python -m app.demo
```

You will be prompted to choose, for example:

- Debate topic  
- Rebuttal rounds  
- Model profiles and other options  

---

## 🚀 Run the Backend Server

Make sure your working directory is the project root, then run:

```bash
uvicorn app.api:app --reload
```

- Default host: `http://127.0.0.1:8000`  
- You can change host/port with `--host` / `--port`.

---

## 💡 Use the Frontend Page

Once `uvicorn` is running, open:

> http://127.0.0.1:8000/

On this simple page, you can:

- Choose the debate mode (AI vs AI / Human vs AI)  
- Choose model profiles and personas  
- Watch the debate stream in real time  

---

## 🧩 Tips for Customization

- In `app/config.py` you can:  
  - Add or edit `MODEL_PROFILES` to connect your own model endpoints.  
  - Customize personas, tone, and responsibilities of each debater.
- In `app/graph.py` you can:  
  - Change debate rounds or add new phases (e.g., free debate).  
- In `index.html` you can:  
  - Replace the minimal UI with any framework you prefer (Vue / React / Svelte / Tailwind, etc.).

---

## ✅ License 

This project is licensed under the MIT License.

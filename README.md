# 多智能体辩论赛（Agent Debate）
Multi‑Agent Debate Demo (FastAPI + LangGraph + LangChain OpenAI)

> 🧪 一个用于展示多智能体「辩论式推理」的 Demo 项目，开箱即用，也方便二次开发。  
> 🧪 A hands-on demo for multi-agent *debate-style reasoning*, easy to run and easy to extend.

---

## ✨ 功能概览 Features

### 🧠 AI vs AI 多智能体辩论  
**AI vs AI Multi-Agent Debate**

- 🤖 裁判 + 正反双方各 4 名辩手（共 9 个智能体）  
  *A judge plus 4 debaters for the Pro side and 4 for the Con side (9 agents in total).*
- 🎭 不同辩位（1–4 辩）分工不同，人设可配置  
  *Each position (1st–4th speaker) has a different role and persona, fully configurable.*
- 🔁 基于 **LangGraph** 的有向图 / FSM 控制辩论流程  
  *Debate flow is orchestrated via a directed graph / FSM built with **LangGraph**.*
- 📡 支持流式输出（NDJSON），便于前端实时展示  
  *Supports **NDJSON streaming** for real-time UI updates.*

---

### 👤 人 vs AI 一对一辩论  
**Human vs AI One-on-One Debate**

- 你可以自由选择站在 **正方 / 反方**  
  *You can freely choose to argue for the **Pro** or **Con** side.*
- AI 队伍由 4 位“辩手”轮流登场：立论 / 驳论 / 举例 / 总结  
  *The AI side cycles through 4 debaters: opening, rebuttal, examples, and summary.*
- 聊天室式交互体验，支持多轮往返辩论  
  *Chat-style interaction with multi-turn debate rounds.*

---

### 🖥 内置简单前端 Simple Built-in Frontend

- 自带一个极简 HTML 页面（`index.html`），无需单独搭建前端工程  
  *Comes with a minimal HTML page (`index.html`), no separate frontend project needed.*
- 通过 `fetch` + NDJSON 流实现「逐句刷新」效果  
  *Uses `fetch` + NDJSON streaming to update debate turns progressively.*
- 默认静态资源（头像等）放在 `app/static/` 下  
  *Default static assets (avatars, etc.) live under `app/static/`.*

---

### 🧩 技术栈 Tech Stack

- **后端 Backend**
  - FastAPI
  - LangGraph
  - LangChain / LangChain OpenAI
- **模型调用 Model Providers**
  - OpenAI / DeepSeek / 通义千问（DashScope）/ Kimi / 智谱 BigModel 等
- **前端 Frontend**
  - HTML + JavaScript

---

## 📁 项目结构 Project Layout

```text
.
├─ requirements.txt
├─ README.md
├─ .env                           # 存放各种模型的 KEY & BASE_URL
└─ app/
   ├─ __init__.py
   ├─ api.py                      # FastAPI 入口 / FastAPI entrypoint
   ├─ agent.py                    # DebateState / AgentRole / speak_with_role
   ├─ graph.py                    # 辩论流程的 LangGraph FSM / LangGraph FSM for debate flow
   ├─ config.py                   # 模型 profile & 预设人格 / model profiles & personas
   ├─ demo.py                     # 命令行 Demo / CLI demo
   ├─ index.html                  # 前端页面 / frontend page
   └─ static/                     # 静态资源：头像等 / static assets (avatars, etc.)
```

---

## 🧱 环境准备 Environment

- ✅ Python 版本 **3.10+**（建议使用虚拟环境）  
  *Python **3.10+** is recommended (ideally inside a virtual environment).*
- ✅ 需要可以访问各大模型提供方的网络环境  
  *Requires network access to the model providers you want to call (OpenAI / DeepSeek / DashScope / Kimi / BigModel, etc.).*

---

## 📦 安装依赖 Installing Dependencies

### 1️⃣ 创建虚拟环境（可选但推荐）  
**Create a virtualenv (optional but recommended)**

```bash
python -m venv .venv
# Windows
# .venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
```

### 2️⃣ 安装依赖 Install requirements

```bash
pip install -r requirements.txt
```

---

## 🔧 配置环境变量 Configure Environment Variables

所有模型配置都在 `app/config.py` 的 `MODEL_PROFILES` 中。  
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

你可以通过环境变量传入所有需要的 KEY 和 BASE_URL：  
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

### 📄 使用 .env 文件（推荐）  
**Recommended: use a `.env` file**

在项目根目录创建 `.env`：  
Create a `.env` file at the project root, for example:

```env
# OpenAI
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.openai.com/v1

# DeepSeek
DEEPSEEK_API_KEY=ds-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 阿里通义（DashScope）
DASHSCOPE_API_KEY=ds-xxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Kimi Moonshot
MOONSHOT_API_KEY=ms-xxxx
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1

# 智谱 BigModel
BIGMODEL_API_KEY=glm-xxxx
BIGMODEL_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

> ⚙️ `config.py` 中已经调用了 `load_dotenv()`，会自动加载 `.env` 中的配置。  
> ⚙️ `load_dotenv()` is already called in `config.py`, so values from `.env` are loaded automatically.

---

## 🖥 命令行 Demo Command-Line Demo

快速在终端体验一场 AI vs AI 辩论：  
Try an AI vs AI debate quickly in your terminal:

```bash
python -m app.demo
```

可以根据提示选择：  
You will be prompted to choose:

- 辩题（topic）  
- 驳论轮次  
- Debate topic  
- Rebuttal round(s)  

---

## 🚀 启动后端服务 Run the Backend Server

确保当前目录在项目根目录：  
Make sure your working directory is the project root, then run:

```bash
uvicorn app.api:app --reload
```

- 默认监听：`http://127.0.0.1:8000`  
  *Default host: `http://127.0.0.1:8000`*
- 可通过 `--host` / `--port` 修改监听地址和端口  
  *You can change host/port with `--host` / `--port`.*

---

## 💡 使用前端页面 Use the Frontend Page

### 访问入口 Entry URL

启动 `uvicorn` 后，在浏览器中打开：  
Once `uvicorn` is running, open:

> http://127.0.0.1:8000/

你将看到一个简单的网页，可以：  
On this simple page, you can:

- 选择辩论模式（AI vs AI / 人 vs AI）  
- 选择使用的模型和人格预设  
- 实时观看辩论内容滚动输出  
- Choose the debate mode (AI vs AI / Human vs AI)  
- Choose model profiles and personas  
- Watch the debate stream in real time

---

## 🧩 二次开发建议 Tips for Customization

- 可以在 `app/config.py` 中：  
  *In `app/config.py` you can:*  
  - 增加 / 修改 `MODEL_PROFILES`，接入你自己的模型服务  
    *Add or edit `MODEL_PROFILES` to connect your own model endpoints;*  
  - 自定义人物设定、口吻、辩位分工等  
    *Customize personas, tone, and responsibilities of each debater.*
- 在 `app/graph.py` 里可以：  
  *In `app/graph.py` you can:*  
  - 修改辩论轮数、流程（例如增加“自由辩论”环节）  
    *Change debate rounds or add new phases (e.g., free debate).*  
- 在 `index.html` 中：  
  *Inside `index.html` you can:*  
  - 替换为任意 UI 框架（Vue / React / Svelte / Tailwind 等）  
    *Swap in any UI framework you prefer (Vue / React / Svelte / Tailwind, etc.).*

---

## ✅ License

本项目采用 MIT 协议开源。  
This project is licensed under the MIT License.


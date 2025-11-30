import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

"""
后端维护一份「模型 profile 表」：
- key 是给前端看的 profile_id
- value 里说明：
    - model: 具体模型名
    - api_key_env: 使用哪一个环境变量里的 key
    - base_url_env: 使用哪一个环境变量里的 base_url（可选）
    - label: 前端展示用名字
    - group: 模型归属厂商 / 分组
"""

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
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


# ========= 预设人格（persona presets） =========

"""
每个预设人格包含：
- label: 前端展示名
- description: 简短说明（给前端展示）
- prompt: 实际注入给大模型的人格描述（中文长描述）
"""

PERSONA_PRESETS: Dict[str, Dict[str, str]] = {
    "neutral_judge": {
        "label": "中立裁判型",
        "description": "理性、冷静、公正，只做中立总结和裁决。",
        "prompt": (
            "你是一位极度理性、公正中立的裁判兼主持人。\n"
            "你不会偏袒任何一方，不会情绪化发言，"
            "会用克制、冷静、客观的语言总结双方观点并给出理性判断。"
        ),
    },
    "calm_logical": {
        "label": "冷静理性分析型",
        "description": "语气平和偏学术风，重视逻辑和结构。",
        "prompt": (
            "你说话冷静克制、逻辑严密，表达风格偏理性与学术化。\n"
            "你会强调论证结构、前提与结论之间的关系，"
            "尽量避免煽情和夸张，用清晰、克制的语言说服对方。"
        ),
    },
    "sharp_attacker": {
        "label": "犀利进攻型",
        "description": "语言犀利直接，重点抓对方漏洞，但仍保持礼貌。",
        "prompt": (
            "你思维敏锐、语言犀利，习惯直接指出对方论证的矛盾和漏洞。\n"
            "在风格上偏进攻型，但始终保持礼貌和尊重，"
            "避免人身攻击，只攻击观点与逻辑。"
        ),
    },
    "humorous": {
        "label": "幽默风趣型",
        "description": "气氛担当，观点里会穿插适度幽默。",
        "prompt": (
            "你表达风格幽默风趣，适度加入轻松、有趣的比喻或吐槽，"
            "让观点更容易被听众接受。\n"
            "虽然风格活泼，但论证本身仍然要有逻辑、有内容，"
            "不能只靠搞笑。"
        ),
    },
    "emotional": {
        "label": "情感共鸣型",
        "description": "善于打比方、讲故事，强调价值和情感。",
        "prompt": (
            "你善于从情感和价值层面说服他人，"
            "喜欢用生活化的例子、故事和比喻来说明观点。\n"
            "你会关注普通人的感受和现实困境，让论证更有人情味。"
        ),
    },
    "data_driven": {
        "label": "数据党型",
        "description": "口吻像做报告，会引用数据/研究（但别瞎编具体数字太多）。",
        "prompt": (
            "你表达风格偏数据和事实导向，喜欢引用统计、研究结论和客观事实。\n"
            "你会使用如“研究表明”“大量案例显示”等表达，"
            "注意不要编造过于具体的数字，保持概括性和可信度。"
        ),
    },
    "philosophical": {
        "label": "哲学思辨型",
        "description": "喜欢抽象、上价值，经常从本质和前提发问。",
        "prompt": (
            "你擅长从本质、前提和价值假设入手进行讨论，"
            "表达风格偏哲学与思辨。\n"
            "你会提出“我们究竟在讨论什么”“这个命题背后的假设是什么”等问题，"
            "帮助听众从更高的抽象层面理解辩题。"
        ),
    },
}


class ModelProfileNotFound(ValueError):
    pass


def create_llm_from_profile(profile_id: str, temperature: float = 1) -> ChatOpenAI:
    """
    根据 profile_id 创建一个 ChatOpenAI 实例。
    供后端自由组合：每个辩手各自用哪个 profile。
    """
    if profile_id not in MODEL_PROFILES:
        raise ModelProfileNotFound(f"Unknown model profile: {profile_id}")

    profile = MODEL_PROFILES[profile_id]

    api_key_env = profile.get("api_key_env")
    base_url_env = profile.get("base_url_env")

    api_key = os.getenv(api_key_env) if api_key_env else None
    base_url = os.getenv(base_url_env) if base_url_env else None

    kwargs: Dict[str, Any] = {
        "model": profile["model"],
        "temperature": temperature,
    }

    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


# 默认使用的 profile（CLI demo 用）
DEFAULT_JUDGE_PROFILE = "deepseek-chat"
DEFAULT_AFF1_PROFILE = "qwen3-max"
DEFAULT_AFF2_PROFILE = "deepseek-reasoner"
DEFAULT_AFF3_PROFILE = "glm-4.5"
DEFAULT_AFF4_PROFILE = "kimi-k2-turbo-preview"
DEFAULT_NEG_PROFILE = "gpt4.1"


def create_default_llms():
    """
    命令行 Demo 用：返回六个默认 LLM。
    （正反双方 4 位辩手默认共用同一个 neg_llm。）
    """
    judge_llm = create_llm_from_profile(DEFAULT_JUDGE_PROFILE)
    aff_llm1 = create_llm_from_profile(DEFAULT_AFF1_PROFILE)
    aff_llm2 = create_llm_from_profile(DEFAULT_AFF2_PROFILE)
    aff_llm3 = create_llm_from_profile(DEFAULT_AFF3_PROFILE)
    aff_llm4 = create_llm_from_profile(DEFAULT_AFF4_PROFILE)
    neg_llm = create_llm_from_profile(DEFAULT_NEG_PROFILE)
    return judge_llm, aff_llm1, aff_llm2, aff_llm3, aff_llm4, neg_llm


def get_model_profiles_meta():
    """
    给前端用的「可选模型列表」元信息。
    可以直接在 API 里返回，让前端渲染下拉框。
    """
    return {
        profile_id: {
            "label": p.get("label", profile_id),
            "model": p.get("model"),
            "group": p.get("group", ""),
        }
        for profile_id, p in MODEL_PROFILES.items()
    }


def get_persona_presets_meta():
    """
    给前端用的「预设人格列表」元信息。
    返回结构：
    {
      "calm_logical": {
        "label": "冷静理性分析型",
        "description": "语气平和偏学术风，重视逻辑和结构。"
      },
      ...
    }
    """
    return {
        preset_id: {
            "label": p.get("label", preset_id),
            "description": p.get("description", ""),
        }
        for preset_id, p in PERSONA_PRESETS.items()
    }

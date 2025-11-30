from __future__ import annotations

import os
import json
from typing import List, Optional, Literal, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    get_model_profiles_meta,
    create_llm_from_profile,
    ModelProfileNotFound,
    get_persona_presets_meta,
    PERSONA_PRESETS,
)
from .graph import build_debate_graph
from .agent import DebateState, AgentRole, speak_with_role

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


# ===== FastAPI 实例 & 静态文件 / 前端挂载 =====

app = FastAPI(title="AI Debate Backend")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_HTML = os.path.join(BASE_DIR, "index.html")

# 挂载静态目录：/static/... 用于头像等资源
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """
    返回前端页面（index.html）。
    这样后端和一个简单的前端就可以放在同一个代码仓里，直接访问 / 即可。
    """
    if not os.path.exists(INDEX_HTML):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_HTML)


# ===== 请求 / 响应模型 =====

class AgentConfig(BaseModel):
    """
    单个辩手 / 裁判配置：
    - profile_id: 使用哪个模型 profile
    - persona_preset_id: 选用哪种预设人格（可选）
    - persona: 自定义性格与语气描述（可选，若提供则优先使用）
    """
    profile_id: str = Field(..., description="对应 config.MODEL_PROFILES 中的 key")
    persona_preset_id: Optional[str] = Field(
        None,
        description="预设人格 ID，见 /personas 接口返回；若同时提供 persona，则 persona 优先",
    )
    persona: Optional[str] = Field(
        None,
        description="自定义性格与语气描述，例如：冷静克制、犀利直接、幽默风趣等（优先级最高）",
    )


class TeamConfig(BaseModel):
    """
    一方 4 名辩手的配置：一辩 / 二辩 / 三辩 / 四辩
    """
    first: AgentConfig
    second: AgentConfig
    third: AgentConfig
    fourth: AgentConfig


class DebateAgentsConfig(BaseModel):
    """
    整场辩论的角色配置：裁判 + 正方 4 人 + 反方 4 人
    """
    judge: AgentConfig
    aff: TeamConfig
    neg: TeamConfig


class DebateMessage(BaseModel):
    role: str
    content: str


class DebateRequest(BaseModel):
    """
    完整 AI vs AI 辩论请求。
    前端可以通过 /models 拿到 profile 列表，通过 /personas 拿到人格列表，
    然后按需填充 agents。
    """
    topic: str = Field(..., description="辩题")
    rounds: int = Field(
        2,
        ge=1,
        le=10,
        description="驳论轮数（正反各一次为一轮）",
    )
    agents: DebateAgentsConfig


class DebateResponse(BaseModel):
    topic: str
    rounds: int
    agents: DebateAgentsConfig
    messages: List[DebateMessage]


# --- 人机一对一辩论（前端轮询调用） ---

class HumanVsAIDebateRequest(BaseModel):
    """
    人类 vs AI 辩论的一轮请求。

    - human_side: 人类站在正方还是反方（aff / neg）
    - ai_side:    AI 队伍是哪一方（正方 aff / 反方 neg），如果不传则自动取 human_side 的反方
    - ai_role:    本轮出场的 AI 辩手名，比如“正方一辩”“反方二辩”（前端已经传了）
    - ai_slot_index: 当前 AI 在队伍中的槽位：0=一辩,1=二辩,2=三辩,3=四辩
    - history:    之前的对话历史（human / ai），用于让模型参考上下文
    """
    topic: str
    human_side: Literal["aff", "neg"] = "aff"

    ai_side: Optional[Literal["aff", "neg"]] = Field(
        None,
        description="AI 所在阵营：'aff'=正方, 'neg'=反方。不传则自动取 human_side 的反方。",
    )
    ai_role: Optional[str] = Field(
        None,
        description="该轮发言的 AI 辩手名，例如：正方一辩 / 反方二辩。",
    )
    ai_slot_index: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description="AI 在队伍中的槽位：0=一辩,1=二辩,2=三辩,3=四辩，用来区分职责。",
    )

    ai_profile_id: str
    ai_persona_preset_id: Optional[str] = Field(
        None,
        description="AI 辩手预设人格 ID，见 /personas；若提供 ai_persona，则 ai_persona 优先",
    )
    ai_persona: Optional[str] = Field(
        None,
        description="AI 辩手性格与语气（中文描述），优先级高于预设人格",
    )
    history: List[DebateMessage] = Field(
        default_factory=list,
        description="人机历史发言，role 一般用 'human' / 'ai'。"
    )


class HumanVsAIDebateResponse(BaseModel):
    topic: str
    human_side: Literal["aff", "neg"]
    ai_role: str
    ai_message: DebateMessage


class HumanVsAIJudgeRequest(BaseModel):
    """
    人类 vs AI 模式下，让裁判做总结 / 裁决的请求。
    前端把整场对话的历史（人类 + AI 发言）丢给裁判即可。
    """
    topic: str
    human_side: Literal["aff", "neg"]
    judge_profile_id: str = Field(
        ...,
        description="裁判使用的模型 profile_id（比如前端选的 judge-model）",
    )
    judge_persona_preset_id: Optional[str] = Field(
        "neutral_judge",
        description="裁判人格预设，默认 neutral_judge",
    )
    judge_persona: Optional[str] = Field(
        None,
        description="自定义裁判风格文案（可选）",
    )
    history: List[DebateMessage] = Field(
        default_factory=list,
        description="整场人机辩论的全部发言（人类和 AI），按时间顺序排列。",
    )


class HumanVsAIJudgeResponse(BaseModel):
    topic: str
    human_side: Literal["aff", "neg"]
    judge_message: DebateMessage


# ===== 工具函数 =====

def _get_llm_cached(profile_id: str, cache: Dict[str, ChatOpenAI]) -> ChatOpenAI:
    """
    简单 LLM 缓存：相同 profile_id 在一场请求里只创建一次。
    """
    if profile_id not in cache:
        cache[profile_id] = create_llm_from_profile(profile_id)
    return cache[profile_id]


def _history_to_langchain(history: List[DebateMessage]):
    """
    将简单的 role/content 列表转成 LangChain 的 HumanMessage / AIMessage，
    方便在人机辩论模式下复用 speak_with_role。
    """
    lc_msgs: List[HumanMessage | AIMessage] = []
    for msg in history:
        role_lower = msg.role.lower()
        if role_lower in ["human", "user", "人类", "用户"]:
            lc_msgs.append(HumanMessage(content=msg.content))
        else:
            lc_msgs.append(AIMessage(content=msg.content))
    return lc_msgs


def resolve_persona_from_config(
        cfg: AgentConfig,
        default_if_none: Optional[str] = None,
) -> Optional[str]:
    """
    人格解析优先级：
    1. cfg.persona（自定义）
    2. cfg.persona_preset_id 对应的预设 prompt
    3. default_if_none（函数调用方传入的默认人格描述）
    """
    if cfg.persona:
        return cfg.persona

    if cfg.persona_preset_id and cfg.persona_preset_id in PERSONA_PRESETS:
        return PERSONA_PRESETS[cfg.persona_preset_id]["prompt"]

    return default_if_none


def resolve_persona_for_human_vs_ai(
        ai_persona: Optional[str],
        ai_persona_preset_id: Optional[str],
        default_if_none: Optional[str],
) -> str:
    """
    人机辩论里用的人格解析（简单版本）：
    1. ai_persona
    2. ai_persona_preset_id 对应预设 prompt
    3. default_if_none
    """
    if ai_persona:
        return ai_persona

    if ai_persona_preset_id and ai_persona_preset_id in PERSONA_PRESETS:
        return PERSONA_PRESETS[ai_persona_preset_id]["prompt"]

    return default_if_none


# ===== 路由 =====

@app.get("/models")
def list_models():
    """
    返回所有可用的模型 profile 列表。
    前端可以用这个接口渲染「裁判 / 各辩手」的模型选择下拉框。
    """
    return get_model_profiles_meta()


@app.get("/personas")
def list_personas():
    """
    返回所有可用的预设人格列表。
    前端可以用这个接口渲染人格选择下拉框。
    """
    return get_persona_presets_meta()


@app.post("/debate", response_model=DebateResponse)
def run_debate(req: DebateRequest):
    """
    启动一场完整的 AI vs AI 多智能体辩论（非流式版本）。
    保留给需要一次性拿完整结果的场景。
    """
    cache: Dict[str, ChatOpenAI] = {}
    try:
        judge_llm = _get_llm_cached(req.agents.judge.profile_id, cache)

        aff_first_llm = _get_llm_cached(req.agents.aff.first.profile_id, cache)
        aff_second_llm = _get_llm_cached(req.agents.aff.second.profile_id, cache)
        aff_third_llm = _get_llm_cached(req.agents.aff.third.profile_id, cache)
        aff_fourth_llm = _get_llm_cached(req.agents.aff.fourth.profile_id, cache)

        neg_first_llm = _get_llm_cached(req.agents.neg.first.profile_id, cache)
        neg_second_llm = _get_llm_cached(req.agents.neg.second.profile_id, cache)
        neg_third_llm = _get_llm_cached(req.agents.neg.third.profile_id, cache)
        neg_fourth_llm = _get_llm_cached(req.agents.neg.fourth.profile_id, cache)
    except ModelProfileNotFound as e:
        raise HTTPException(status_code=400, detail=str(e))

    judge_default_persona = (
        PERSONA_PRESETS.get("neutral_judge", {}).get("prompt")
        or "理性、中立、公正的主持人，严格避免偏袒任何一方。"
    )
    judge = AgentRole(
        name="裁判",
        llm=judge_llm,
        persona=resolve_persona_from_config(req.agents.judge, judge_default_persona),
    )

    aff_team: List[AgentRole] = [
        AgentRole(
            "正方一辩",
            aff_first_llm,
            resolve_persona_from_config(req.agents.aff.first),
        ),
        AgentRole(
            "正方二辩",
            aff_second_llm,
            resolve_persona_from_config(req.agents.aff.second),
        ),
        AgentRole(
            "正方三辩",
            aff_third_llm,
            resolve_persona_from_config(req.agents.aff.third),
        ),
        AgentRole(
            "正方四辩",
            aff_fourth_llm,
            resolve_persona_from_config(req.agents.aff.fourth),
        ),
    ]

    neg_team: List[AgentRole] = [
        AgentRole(
            "反方一辩",
            neg_first_llm,
            resolve_persona_from_config(req.agents.neg.first),
        ),
        AgentRole(
            "反方二辩",
            neg_second_llm,
            resolve_persona_from_config(req.agents.neg.second),
        ),
        AgentRole(
            "反方三辩",
            neg_third_llm,
            resolve_persona_from_config(req.agents.neg.third),
        ),
        AgentRole(
            "反方四辩",
            neg_fourth_llm,
            resolve_persona_from_config(req.agents.neg.fourth),
        ),
    ]

    app_graph = build_debate_graph(judge, aff_team, neg_team)

    init_state: DebateState = {
        "topic": req.topic,
        "round": 1,
        "max_rounds": req.rounds,
        "messages": [],
    }

    final_state = app_graph.invoke(init_state)

    messages: List[DebateMessage] = []
    for msg in final_state["messages"]:
        role = getattr(msg, "name", None)
        if not role:
            role = {
                "human": "用户",
                "system": "系统",
            }.get(msg.type, "AI")
        messages.append(DebateMessage(role=role, content=str(msg.content)))

    return DebateResponse(
        topic=req.topic,
        rounds=req.rounds,
        agents=req.agents,
        messages=messages,
    )


@app.post("/debate/stream")
def run_debate_stream(req: DebateRequest):
    """
    流式版本的辩论接口：
    按行输出 NDJSON，每一行是一条消息：
    {"type": "message", "role": "正方一辩", "content": "...", "side": "aff"}
    最后一行是 {"type": "end"} 表示结束。
    """
    cache: Dict[str, ChatOpenAI] = {}
    try:
        judge_llm = _get_llm_cached(req.agents.judge.profile_id, cache)

        aff_first_llm = _get_llm_cached(req.agents.aff.first.profile_id, cache)
        aff_second_llm = _get_llm_cached(req.agents.aff.second.profile_id, cache)
        aff_third_llm = _get_llm_cached(req.agents.aff.third.profile_id, cache)
        aff_fourth_llm = _get_llm_cached(req.agents.aff.fourth.profile_id, cache)

        neg_first_llm = _get_llm_cached(req.agents.neg.first.profile_id, cache)
        neg_second_llm = _get_llm_cached(req.agents.neg.second.profile_id, cache)
        neg_third_llm = _get_llm_cached(req.agents.neg.third.profile_id, cache)
        neg_fourth_llm = _get_llm_cached(req.agents.neg.fourth.profile_id, cache)
    except ModelProfileNotFound as e:
        raise HTTPException(status_code=400, detail=str(e))

    judge_default_persona = (
        PERSONA_PRESETS.get("neutral_judge", {}).get("prompt")
        or "理性、中立、公正的主持人，严格避免偏袒任何一方。"
    )
    judge = AgentRole(
        name="裁判",
        llm=judge_llm,
        persona=resolve_persona_from_config(req.agents.judge, judge_default_persona),
    )

    aff_team: List[AgentRole] = [
        AgentRole(
            "正方一辩",
            aff_first_llm,
            resolve_persona_from_config(req.agents.aff.first),
        ),
        AgentRole(
            "正方二辩",
            aff_second_llm,
            resolve_persona_from_config(req.agents.aff.second),
        ),
        AgentRole(
            "正方三辩",
            aff_third_llm,
            resolve_persona_from_config(req.agents.aff.third),
        ),
        AgentRole(
            "正方四辩",
            aff_fourth_llm,
            resolve_persona_from_config(req.agents.aff.fourth),
        ),
    ]

    neg_team: List[AgentRole] = [
        AgentRole(
            "反方一辩",
            neg_first_llm,
            resolve_persona_from_config(req.agents.neg.first),
        ),
        AgentRole(
            "反方二辩",
            neg_second_llm,
            resolve_persona_from_config(req.agents.neg.second),
        ),
        AgentRole(
            "反方三辩",
            neg_third_llm,
            resolve_persona_from_config(req.agents.neg.third),
        ),
        AgentRole(
            "反方四辩",
            neg_fourth_llm,
            resolve_persona_from_config(req.agents.neg.fourth),
        ),
    ]

    app_graph = build_debate_graph(judge, aff_team, neg_team)

    init_state: DebateState = {
        "topic": req.topic,
        "round": 1,
        "max_rounds": req.rounds,
        "messages": [],
    }

    def event_gen():
        last_len = 0
        try:
            # 关键：用 LangGraph 的流式执行
            for state in app_graph.stream(init_state, stream_mode="values"):
                msgs = state["messages"]
                # 只把“新增”的消息发出去
                for msg in msgs[last_len:]:
                    role = getattr(msg, "name", None)
                    if not role:
                        role = {
                            "human": "用户",
                            "system": "系统",
                        }.get(msg.type, "AI")

                    if "正方" in role:
                        side = "aff"
                    elif "反方" in role:
                        side = "neg"
                    elif role == "裁判":
                        side = "judge"
                    else:
                        side = "judge"

                    chunk = {
                        "type": "message",
                        "role": role,
                        "content": str(msg.content),
                        "side": side,
                    }
                    yield json.dumps(chunk, ensure_ascii=False) + "\n"

                last_len = len(msgs)

        except Exception as exception:
            err = {"type": "error", "detail": str(exception)}
            yield json.dumps(err, ensure_ascii=False) + "\n"
        finally:
            # 通知前端结束
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/debate/human-vs-ai", response_model=HumanVsAIDebateResponse)
def human_vs_ai_debate(req: HumanVsAIDebateRequest):
    """
    人类 vs AI 一对一辩论模式的一轮发言。
    支持按一辩 / 二辩 / 三辩 / 四辩区分不同职责和提示词。
    """
    try:
        ai_llm = create_llm_from_profile(req.ai_profile_id)
    except ModelProfileNotFound as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 1. 先确定 AI 所在阵营：
    #    优先用前端传的 ai_side，没传则默认“与人类相反”
    if req.ai_side is not None:
        ai_side: Literal["aff", "neg"] = req.ai_side
    else:
        ai_side = "neg" if req.human_side == "aff" else "aff"

    side_cn = "正方" if ai_side == "aff" else "反方"
    numerals = ["一", "二", "三", "四"]

    # 2. 角色名优先用前端传过来的 ai_role，其次根据阵营 + 辩位推一个
    if req.ai_role:
        ai_role_name = req.ai_role
    elif req.ai_slot_index is not None and 0 <= req.ai_slot_index < 4:
        ai_role_name = f"{side_cn}{numerals[req.ai_slot_index]}辩"
    else:
        # 兜底：老版本只有“正方辩手 / 反方辩手”
        ai_role_name = f"{side_cn}辩手"

    # 3. 默认人格：根据阵营 + 辩位给更细致的描述，再套原来的优先级逻辑
    default_persona = _default_persona_for_slot(
        ai_side=ai_side,
        slot_index=req.ai_slot_index,
    )
    persona = resolve_persona_for_human_vs_ai(
        ai_persona=req.ai_persona,
        ai_persona_preset_id=req.ai_persona_preset_id,
        default_if_none=default_persona,
    )

    agent = AgentRole(name=ai_role_name, llm=ai_llm, persona=persona)

    # 4. 把前端传来的 history 转成 LangChain 的消息
    history_msgs = _history_to_langchain(req.history)

    state: DebateState = {
        "topic": req.topic,
        "round": 1,
        "max_rounds": 1,
        "messages": history_msgs,
    }

    # 5. 根据“第几辩”构造不同的指令 prompt
    instructions = _build_slot_instruction(
        ai_role_name=ai_role_name,
        slot_index=req.ai_slot_index,
        history_len=len(req.history),
    )

    # 6. 让对应“辩手”说话
    result = speak_with_role(agent, instructions, state)
    ai_msg = result["messages"][-1]

    return HumanVsAIDebateResponse(
        topic=req.topic,
        human_side=req.human_side,
        ai_role=ai_role_name,
        ai_message=DebateMessage(role=ai_role_name, content=str(ai_msg.content)),
    )


@app.post("/debate/human-vs-ai/judge", response_model=HumanVsAIJudgeResponse)
def human_vs_ai_judge(req: HumanVsAIJudgeRequest):
    """
    人类 vs AI 模式下，让裁判对整场辩论做一次总结 + 裁决。
    风格尽量贴近 AI vs AI 模式里的裁判总结。
    """
    try:
        judge_llm = create_llm_from_profile(req.judge_profile_id)
    except ModelProfileNotFound as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 裁判人格：自定义 > 预设 neutral_judge > 默认兜底
    default_judge_persona = (
        PERSONA_PRESETS.get("neutral_judge", {}).get("prompt")
        or "你是一位理性、中立、公正的裁判兼主持人，只负责客观总结与评判。"
    )
    if req.judge_persona:
        persona = req.judge_persona
    elif req.judge_persona_preset_id and req.judge_persona_preset_id in PERSONA_PRESETS:
        persona = PERSONA_PRESETS[req.judge_persona_preset_id]["prompt"]
    else:
        persona = default_judge_persona

    judge = AgentRole(name="裁判", llm=judge_llm, persona=persona)

    history_msgs = _history_to_langchain(req.history)

    state: DebateState = {
        "topic": req.topic,
        "round": 1,
        "max_rounds": 1,
        "messages": history_msgs,
    }

    instructions = (
        "当前阶段：裁判总结与裁决。\n"
        "你已经完整看完了本场【人类 vs AI】辩论从开场到现在的全部发言，"
        "请严格基于实际发言内容来做评判，不要凭空想象双方说过但实际没有说过的内容。\n\n"
        "请你以现场口头总结发言的形式完成下面的内容，语言自然连贯，像站在台上面对观众说话：\n"
        "1）先用一两句话简单回顾辩题，以及人类一方和 AI 一方各自的大致立场；\n"
        "2）从“观点是否清晰成体系、论证逻辑是否严密、论据是否有说服力、攻防环节表现如何”等维度，"
        "分别评价两方的整体表现，指出各自做得好的地方和明显的不足；\n"
        "3）综合以上维度，明确说出你认为哪一方整体上更具说服力，并用几句话说明你的裁决依据；\n"
        "4）最后用一小段理性、中立的结语，提醒大家这只是一次基于现场表现的评判，"
        "目的是促进思考和观点碰撞，而不是给这个辩题下唯一标准答案。\n\n"
        "注意：\n"
        "• 不要输出任何 markdown 标记、列表或小标题，只用 1~3 段连续的自然段来表达；\n"
        "• 不要长篇复述辩手原话，重在概括和评价，而不是转录。"
    )

    result = speak_with_role(judge, instructions, state)
    judge_msg = result["messages"][-1]

    return HumanVsAIJudgeResponse(
        topic=req.topic,
        human_side=req.human_side,
        judge_message=DebateMessage(
            role="裁判",
            content=str(judge_msg.content),
        ),
    )


# ===== Human vs AI 辅助：根据阵营 + 槽位给默认人格 & 指令 =====

def _default_persona_for_slot(ai_side: str, slot_index: Optional[int]) -> str:
    """
    根据 AI 所在阵营 + 辩位，给一个更细的默认人格描述：
    一辩：框架清晰的立论
    二辩：犀利驳论
    三辩：举例补充 / 较活泼
    四辩：沉稳总结
    """
    side_cn = "正方" if ai_side == "aff" else "反方"

    if slot_index == 0:
        return f"冷静、条理清晰的{side_cn}一辩，负责开篇立论，结构严谨，善于搭建整体框架。"
    if slot_index == 1:
        return f"逻辑敏锐、善于抓漏洞的{side_cn}二辩，主要负责针对对方进行驳论并巩固本方论证。"
    if slot_index == 2:
        return f"幽默机智、善于举例说明的{side_cn}三辩，负责用生动案例和类比支持本方观点，并适度补充反驳。"
    if slot_index == 3:
        return f"沉稳有气场的{side_cn}四辩，负责总结全场、提炼本方优势论点并进行收束。"

    # 兜底：保持你之前的默认人格
    if ai_side == "neg":
        return "逻辑缜密、反应迅速的反方辩手，善于抓住对方论证中的漏洞。"
    else:
        return "立场坚定、善于构建体系化论证的正方辩手，语言有气势但保持礼貌。"


def _build_slot_instruction(
        ai_role_name: str,
        slot_index: Optional[int],
        history_len: int,
) -> str:
    """
    根据当前是第几辩，给不同职责的指令 prompt。
    如果 slot_index 为空，则回退到老的“首轮立论 / 之后驳论”逻辑。
    """
    # 一辩：开篇立论，不强调逐点反驳
    if slot_index == 0:
        return (
            f"你现在扮演 {ai_role_name}，是本方的一辩，主要负责【开篇立论】。\n"
            "请在本轮发言中：\n"
            "1）明确表明本方在该辩题下的立场；\n"
            "2）给出 2~3 个结构清晰的核心论点，可以适度点到论据；\n"
            "3）可以预判对方可能的质疑，但不要把重点放在逐条反驳对方，而是搭建本方完整的论证框架；\n"
            "4）语言要条理清晰、便于后续队友沿用。"
        )

    # 二辩：主力驳论
    if slot_index == 1:
        return (
            f"你现在扮演 {ai_role_name}，是本方的二辩，主要负责【驳论与巩固本方论证】。\n"
            "请在本轮发言中：\n"
            "1）先简要概括对方最近一轮发言的关键观点；\n"
            "2）逐点指出其中的逻辑漏洞、前提问题或证据不足；\n"
            "3）补充或强化本方 1~2 个关键论点（可引用一辩提出的框架）；\n"
            "4）逻辑要严密，语气可以犀利但保持礼貌。"
        )

    # 三辩：举例 & 侧翼补充
    if slot_index == 2:
        return (
            f"你现在扮演 {ai_role_name}，是本方的三辩，主要负责【举例补充与侧翼支持】。\n"
            "请在本轮发言中：\n"
            "1）选取 1~2 个有代表性的案例、比喻或类比，从新的角度支持本方核心论点；\n"
            "2）可以适度回应对方观点，但重点是用生动具体的例子让观众更容易理解并记住本方立场；\n"
            "3）语言可以稍微活泼、有一点幽默感，但不要喧宾夺主、避免跑题。"
        )

    # 四辩：总结陈词
    if slot_index == 3:
        return (
            f"你现在扮演 {ai_role_name}，是本方的四辩，主要负责【总结陈词】。\n"
            "请在本轮发言中：\n"
            "1）高度概括本方此前的核心论点（不要细节重复，重在提炼与对比）；\n"
            "2）点出对方论证中的关键问题，说明其不足以推翻本方立场；\n"
            "3）给出整体性的、具有说服力的总结和价值判断，可以略带煽动性但保持理性；\n"
            "4）语言要有收束感，让裁判和观众清楚记住本方的立场和优势。"
        )

    # ⭐ 兜底逻辑：保持原来的“首轮立论 / 后续驳论”
    if history_len == 0:
        return (
            "当前是本场辩论的第一轮发言，你需要先给出本方的立场和立论：\n"
            "1）明确表明你在该辩题下的立场；\n"
            "2）给出 2~3 个核心论点，条理清晰；\n"
            "3）可以适度预判对方可能的质疑，但重点在于正面展开本方观点。"
        )
    else:
        return (
            "你正在和一名人类辩手进行一对一中文辩论。\n"
            "请在本轮发言中：\n"
            "1）紧扣对方最近一轮发言进行回应和反驳（先简要复述对方要点）；\n"
            "2）指出对方论证中的漏洞、前提问题或不充分之处；\n"
            "3）补充或强化本方的 1~2 个关键论点；\n"
            "4）保持礼貌和专业，不要进行人身攻击。"
        )

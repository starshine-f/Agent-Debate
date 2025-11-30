from dataclasses import dataclass
from typing import Optional

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


class DebateState(MessagesState):
    """
    LangGraph 的 State：
    - 继承 MessagesState，自动包含 messages 列表
    - 再加上我们自定义的 topic / round / max_rounds
    """
    topic: str       # 辩题
    round: int       # 当前驳论轮数（从 1 开始）
    max_rounds: int  # 总共需要的驳论轮数


@dataclass
class AgentRole:
    """
    单个角色（裁判 / 正方一辩 / 反方三辩等）的配置：
    - name: 在对话中展示的角色名称
    - llm: 该角色背后的大模型实例
    - persona: 性格与语气描述（可选）
    """
    name: str
    llm: ChatOpenAI
    persona: Optional[str] = None


def speak_with_role(
    agent: AgentRole,
    instructions: str,
    state: DebateState,
):
    """
    通用发言封装：
    - 用 system prompt 塞角色 + 辩题 + 角色性格 + 任务说明
    - 使用当前 state["messages"] 作为历史记录
    - 返回 {"messages": [AIMessage(...)]} 方便 LangGraph 合并
    """
    topic = state["topic"]
    history = state["messages"]

    # 1. 身份设定（基础信息）
    system_lines: list[str] = [
        "【身份设定】",
        f"你现在扮演中文辩论赛中的「{agent.name}」。",
        f"本场辩题是：{topic}",
        "",
    ]

    # 2. 人格 / 风格设定（如果有）
    if agent.persona:
        system_lines.append("【人格 / 风格设定】")
        system_lines.append(agent.persona)
        system_lines.append(
            "在整个辩论过程中，你必须始终保持上述人设的一致性，"
            "在语气、用词、句式、比喻和论证方式上都要主动向该风格靠拢。"
        )
        system_lines.append(
            "当「任务要求」和「人设风格」发生冲突时，优先保证表达风格符合人设，"
            "在不违背事实的前提下可以适度调整内容以贴合人设。"
        )
        system_lines.append(
            "你可以通过用词选择、段落节奏、比喻方式、情绪强度等手段，"
            "让听众一眼就能感受到你的性格特点。"
        )
        system_lines.append("")

    # 3. 当前任务说明
    system_lines.append("【当前任务】")
    system_lines.append(instructions)
    system_lines.append("")

    # 4. 输出要求
    system_lines.append("【输出要求】")
    system_lines.append("1）使用简体中文；")
    system_lines.append("2）不要说明你在调用大模型，也不要解释规则，只给出辩论内容；")
    system_lines.append("3）不超过 400 字；")
    system_lines.append("4）在语气与行文风格上，要明显体现前述人格特征；")
    system_lines.append("5）内容要有清晰的逻辑结构，避免纯情绪输出或空洞口号。")

    system = SystemMessage(content="\n".join(system_lines))

    messages = [system] + list(history) + [
        HumanMessage(content="请继续本轮辩论，根据上方规则生成你的发言。")
    ]
    resp: AIMessage = agent.llm.invoke(messages)

    # 在 LangGraph 的 state 里，通过 name 标记是哪个辩手说的话
    resp.name = agent.name
    return {"messages": [resp]}


__all__ = [
    "DebateState",
    "AgentRole",
    "speak_with_role",
]

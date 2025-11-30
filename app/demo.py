from .graph import build_debate_graph
from .agent import DebateState, AgentRole
from .config import create_default_llms, PERSONA_PRESETS


def pick_preset(preset_id: str, fallback: str) -> str:
    """
    从 PERSONA_PRESETS 里取对应预设的人格 prompt。
    如果找不到这个 preset_id，就用 fallback 文案兜底。
    """
    preset = PERSONA_PRESETS.get(preset_id)
    if not preset:
        return fallback
    return preset.get("prompt", fallback)


def run_demo():
    topic = input("请输入本场辩题：").strip()
    if not topic:
        topic = "在基础教育阶段，应更注重培养学生的知识广度，还是学科深度？"

    rounds_str = input("请输入驳论轮数（默认 2）：").strip()
    max_rounds = int(rounds_str) if rounds_str else 2

    # 默认的 profile 创建 LLM
    judge_llm, aff_llm1, aff_llm2, aff_llm3, aff_llm4, neg_llm = create_default_llms()

    # ===== 使用预设人格：裁判 + 正方/反方四个辩手 =====

    judge_persona = pick_preset(
        "neutral_judge",
        "理性克制、中立公正的主持人，负责介绍流程并做总结裁决。",
    )
    judge = AgentRole(
        name="裁判",
        llm=judge_llm,
        persona=judge_persona,
    )

    # 正方：一辩偏「冷静理性」，二辩偏「犀利进攻」，三辩带点「幽默」，四辩偏「情感总结」
    aff_team = [
        AgentRole(
            "正方一辩",
            aff_llm1,
            pick_preset(
                "calm_logical",
                "结构清晰、逻辑严谨，擅长搭建论证框架。",
            ),
        ),
        AgentRole(
            "正方二辩",
            aff_llm2,
            pick_preset(
                "sharp_attacker",
                "反应敏捷，善于针对对方观点提问和追问。",
            ),
        ),
        AgentRole(
            "正方三辩",
            aff_llm3,
            pick_preset(
                "humorous",
                "语言风趣，适度幽默，但保持内容有逻辑。",
            ),
        ),
        AgentRole(
            "正方四辩",
            aff_llm4,
            pick_preset(
                "emotional",
                "擅长从情感和价值层面做总结与升华。",
            ),
        ),
    ]

    # 反方：一辩偏「哲学思辨」，二辩偏「数据党」，三辩默认理性，四辩做全局总结
    neg_team = [
        AgentRole(
            "反方一辩",
            neg_llm,
            pick_preset(
                "philosophical",
                "冷静理性，善于质疑命题的前提与假设。",
            ),
        ),
        AgentRole(
            "反方二辩",
            neg_llm,
            pick_preset(
                "data_driven",
                "思路活跃，喜欢用事实和数据拆解对方论证。",
            ),
        ),
        AgentRole(
            "反方三辩",
            neg_llm,
            pick_preset(
                "calm_logical",
                "表达有条理，负责补充和梳理反方论证细节。",
            ),
        ),
        AgentRole(
            "反方四辩",
            neg_llm,
            pick_preset(
                "calm_logical",
                "总结能力强，擅长从全局视角比较双方立场。",
            ),
        ),
    ]

    app = build_debate_graph(judge, aff_team, neg_team)

    # 初始化 State
    init_state: DebateState = {
        "topic": topic,
        "round": 1,
        "max_rounds": max_rounds,
        "messages": [],
    }

    # 每个节点执行完就输出这一阶段的发言
    stage_labels = {
        "intro": "【开场】",
        "opening_aff": "【正方一辩立论】",
        "opening_neg": "【反方一辩立论】",
        "refute_aff": "【正方驳论】",
        "refute_neg": "【反方驳论】",
        "closing_aff": "【正方总结】",
        "closing_neg": "【反方总结】",
        "judge_summary": "【裁判总结】",
    }

    print("\n==================== 辩论开始 ====================\n")

    # 使用 stream，而不是 invoke
    for event in app.stream(init_state):
        # event 是一个 dict：{node_name: state_after_this_node}
        for node_name, node_state in event.items():
            # LangGraph 在结束时会给一个 "__end__" 事件，这里可以跳过
            if node_name.startswith("__"):
                continue

            messages = node_state.get("messages", [])
            if not messages:
                continue

            last = messages[-1]
            role = getattr(last, "name", None)
            if not role:
                role = {
                    "human": "用户",
                    "system": "系统",
                }.get(getattr(last, "type", ""), "AI")

            stage = stage_labels.get(node_name, f"【{node_name}】")
            print(stage)
            print(f"[{role}]\n{last.content}\n")

    print("==================== 辩论结束 ====================\n")


if __name__ == "__main__":
    run_demo()

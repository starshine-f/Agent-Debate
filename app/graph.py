from typing import List
from langgraph.graph import StateGraph, END

from .agent import DebateState, AgentRole, speak_with_role


# ===== 辅助函数：根据轮次选择哪位辩手负责驳论 =====

def pick_refuter(team: List[AgentRole], round_num: int) -> AgentRole:
    """
    简单规则：
    - 第 1 轮驳论：二辩
    - 第 2 轮驳论：三辩
    - 第 3 轮及之后：四辩
    如果对应位置不存在（极端情况），就 fallback 到队伍最后一名。
    """
    if round_num <= 1 and len(team) >= 2:
        return team[1]  # 二辩
    if round_num == 2 and len(team) >= 3:
        return team[2]  # 三辩
    if len(team) >= 4:
        return team[3]  # 四辩
    return team[-1]  # 兜底


# ===== 各个阶段的节点函数，接受 AgentRole 作为“闭包”变量 =====

def make_intro_node(judge: AgentRole):
    def intro_node(state: DebateState):
        """INTRO：裁判开场白"""
        instructions = (
            "你的任务是以裁判兼主持人的身份，用 3~5 句话介绍："
            "1）本场辩题内容；2）正反双方立场；3）大致流程（立论-驳论-总结-裁决）。\n"
            "注意保持中立，不要替任意一方辩护。"
        )
        return speak_with_role(judge, instructions, state)

    return intro_node


def make_opening_aff_node(aff_first: AgentRole):
    def opening_aff_node(state: DebateState):
        """OPENING_AFF：正方一辩立论"""
        instructions = (
            "当前阶段：正方一辩立论。\n"
            "请作为支持该命题的一辩，系统地提出你的立场和核心论据，"
            "可以用条目列出 2~4 个关键论点，暂时不需要攻击对方，只做正面论证。"
        )
        return speak_with_role(aff_first, instructions, state)

    return opening_aff_node


def make_opening_neg_node(neg_first: AgentRole):
    def opening_neg_node(state: DebateState):
        """OPENING_NEG：反方一辩立论"""
        instructions = (
            "当前阶段：反方一辩立论。\n"
            "你反对该命题，请提出与你立场相反的主要论点，同样用 2~4 个关键理由展开，"
            "不要直接反驳正方，只要清晰表达自己的立场和逻辑。"
        )
        return speak_with_role(neg_first, instructions, state)

    return opening_neg_node


def make_refute_aff_node(aff_team: List[AgentRole]):
    def refute_aff_node(state: DebateState):
        """REFUTE_AFF：正方驳对方 + 补充论据"""
        cur_round = state["round"]
        speaker = pick_refuter(aff_team, cur_round)
        instructions = (
            f"当前阶段：第 {cur_round} 轮驳论（正方回合），由你代表正方发言。\n"
            "请重点围绕对方【最近一次发言】进行反驳：\n"
            "1）先简要复述对方的关键观点；\n"
            "2）指出其中的漏洞或前提问题；\n"
            "3）补充 1~2 条新的论据强化正方立场。\n"
            "注意保持礼貌、逻辑清晰，不要人身攻击。"
        )
        return speak_with_role(speaker, instructions, state)

    return refute_aff_node


def make_refute_neg_node(neg_team: List[AgentRole]):
    def refute_neg_node(state: DebateState):
        """REFUTE_NEG：反方驳对方 + 补充论据，并在这里递增轮数"""
        cur_round = state["round"]
        speaker = pick_refuter(neg_team, cur_round)
        instructions = (
            f"当前阶段：第 {cur_round} 轮驳论（反方回合），由你代表反方发言。\n"
            "请重点围绕正方【最近一次发言】进行反驳：\n"
            "1）先简要复述正方的关键观点；\n"
            "2）指出其中的漏洞或不充分之处；\n"
            "3）补充 1~2 条新的论据强化反方立场。\n"
            "注意逻辑严谨，可以适度引用常识或经验，但不要捏造具体数据。"
        )
        result = speak_with_role(speaker, instructions, state)

        # 完成一整轮（正方+反方）后，轮数 +1
        new_round = state.get("round", 1) + 1
        result["round"] = new_round
        return result

    return refute_neg_node


def make_closing_aff_node(aff_fourth: AgentRole):
    def closing_aff_node(state: DebateState):
        """CLOSING_AFF：正方四辩总结"""
        instructions = (
            "当前阶段：正方总结陈词，由正方四辩完成。\n"
            "请用 2~4 段话完成：\n"
            "1）归纳本方最核心的 2~3 个论点；\n"
            "2）简要指出你认为对方观点中最致命的 1~2 个问题；\n"
            "3）给出一句有力的收尾金句，强化听众对正方立场的记忆。"
        )
        return speak_with_role(aff_fourth, instructions, state)

    return closing_aff_node


def make_closing_neg_node(neg_fourth: AgentRole):
    def closing_neg_node(state: DebateState):
        """CLOSING_NEG：反方四辩总结"""
        instructions = (
            "当前阶段：反方总结陈词，由反方四辩完成。\n"
            "请用 2~4 段话完成：\n"
            "1）归纳本方最核心的 2~3 个论点；\n"
            "2）简要指出你认为正方观点中最关键的漏洞；\n"
            "3）给出一句具有说服力的收尾陈述，总结为什么听众更应该接受反方立场。"
        )
        return speak_with_role(neg_fourth, instructions, state)

    return closing_neg_node


def make_judge_summary_node(judge: AgentRole):
    def judge_summary_node(state: DebateState):
        """JUDGE_SUMMARY：裁判给出结论（AI vs AI 模式用）"""
        instructions = (
            "当前阶段：裁判总结与裁决。\n"
            "你已经完整看完了本场辩论从开场到结束的全部对话，"
            "请严格基于实际发言内容来做评判，不要凭空想象双方说过但实际没有说过的内容。\n\n"
            "请你以现场口头总结发言的形式完成下面的内容，语言自然连贯，像站在台上面对观众说话：\n"
            "1）先用一两句话简单回顾辩题，以及正反双方各自的大致立场；\n"
            "2）分别从“角色完成度”的角度，点评正方四位辩手和反方四位辩手："
            "结合他们各自应该承担的典型任务（例如：一辩负责界定与立论、二辩偏进攻与质询、"
            "三辩偏总结与补充论点、四辩偏收束与升华），说明他们在实际发言中做得好的地方和明显的不足；\n"
            "3）在此基础上，从“队伍整体”的视角，评价正反双方各自的整体表现："
            "包括论点是否清晰成体系、团队内部是否呼应配合、论证是否自洽、是否紧扣辩题；\n"
            "4）专门就攻防环节（相互质询和回应对方观点的部分）做一个小结："
            "指出哪一方在提出攻击点、追问细节、抓对方漏洞方面更有亮点，"
            "哪一方在回应质疑、化解攻击、稳住己方立场方面做得更好，并给出简要理由；\n"
            "5）综合以上个人表现、队伍整体表现以及攻防环节的对比，明确说出你认为哪一方整体上更有说服力，"
            "并用几句话解释你的裁决依据（可以提到论证深度、逻辑严密性、论据质量、回应有效性等维度）；\n"
            "6）最后用一小段理性、中立的结语，提醒大家这只是一次基于当场表现的评判，"
            "目的是促进思考和观点碰撞，而不是给出这个辩题的唯一标准答案。\n\n"
            "注意：\n"
            "• 整段发言不要使用任何小标题、编号列表、项目符号或加粗等排版格式；\n"
            "• 不要输出 markdown，只用一到三段连续的自然段来表达，好像你在现场做口头总结一样；\n"
            "• 不要逐字复述辩手原话，重点是概括和评价，而不是转录。"
        )
        return speak_with_role(judge, instructions, state)

    return judge_summary_node


# ===== 路由函数：决定是否继续下一轮驳论 =====

def route_after_refute_neg(state: DebateState) -> str:
    """
    在反方驳论结束后，根据当前轮数决定：
    - 若还没达到 max_rounds：回到 refute_aff（下一轮）
    - 否则：进入 closing_aff（双方总结阶段）
    """
    if state["round"] <= state["max_rounds"]:
        return "refute_aff"
    return "closing_aff"


# ===== 构建 LangGraph 图（状态机），接收裁判 + 正反双方 4 名辩手 =====

def build_debate_graph(
    judge: AgentRole,
    aff_team: List[AgentRole],
    neg_team: List[AgentRole],
):
    """
    FSM 流程：

    INTRO -> OPENING_AFF -> OPENING_NEG
           -> (REFUTE_AFF -> REFUTE_NEG) * k
           -> CLOSING_AFF -> CLOSING_NEG -> JUDGE_SUMMARY -> END
    """
    if len(aff_team) < 1 or len(neg_team) < 1:
        raise ValueError("aff_team / neg_team 至少需要 1 名辩手")

    # 约定：team[0] = 一辩, team[1] = 二辩, ...
    aff_first = aff_team[0]
    neg_first = neg_team[0]
    aff_fourth = aff_team[3] if len(aff_team) >= 4 else aff_team[-1]
    neg_fourth = neg_team[3] if len(neg_team) >= 4 else neg_team[-1]

    builder = StateGraph(DebateState)

    # 把 agent 通过闭包塞进各个节点
    builder.add_node("intro", make_intro_node(judge))
    builder.add_node("opening_aff", make_opening_aff_node(aff_first))
    builder.add_node("opening_neg", make_opening_neg_node(neg_first))
    builder.add_node("refute_aff", make_refute_aff_node(aff_team))
    builder.add_node("refute_neg", make_refute_neg_node(neg_team))
    builder.add_node("closing_aff", make_closing_aff_node(aff_fourth))
    builder.add_node("closing_neg", make_closing_neg_node(neg_fourth))
    builder.add_node("judge_summary", make_judge_summary_node(judge))

    # 起点
    builder.set_entry_point("intro")

    # 固定顺序的边
    builder.add_edge("intro", "opening_aff")
    builder.add_edge("opening_aff", "opening_neg")
    builder.add_edge("opening_neg", "refute_aff")
    builder.add_edge("refute_aff", "refute_neg")

    # 条件边：多轮驳论
    builder.add_conditional_edges("refute_neg", route_after_refute_neg)

    # 收尾阶段
    builder.add_edge("closing_aff", "closing_neg")
    builder.add_edge("closing_neg", "judge_summary")
    builder.add_edge("judge_summary", END)

    return builder.compile()

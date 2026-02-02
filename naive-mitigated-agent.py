# bounded_langgraph_agent.py
"""
MITIGATED version that prevents Coherence Decay via context compression.

Key difference: After each tool call, we COMPRESS the output into a structured
memory store, then DISCARD the raw tool output from the message history.

This keeps GPU KV cache bounded regardless of how many PDFs/simulations we run.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, TypedDict, TYPE_CHECKING

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from common import make_llm, TOOLS, fetch_annual_report, run_monte_carlo, quick_math, logger, set_progress, progress_bar


# -------------------------
# Agent State
# -------------------------
class AgentState(TypedDict):
    messages: List[AnyMessage]              # full conversation
    memory: str                             # compressed running memory
    i: int                                  # iteration counter
    urls: List[str]                         # planned URLs


SYSTEM = """You are a Financial Quant Agent with BOUNDED CONTEXT.
Rules:
- Use tools to fetch annual reports and run simulations.
- After each tool call, your memory will be COMPRESSED.
- The compressed memory contains ONLY the key facts you need.
- Never try to recall raw document text - it's been compacted.
- Build your final report from the compressed memory only.
"""


def planner_node(llm: ChatOpenAI, tools: list):
    """
    Decide what to do next: call tool vs finish.
    We *do not* pass huge tool outputs; we pass only the compressed memory.
    """
    llm_with_tools = llm.bind_tools(tools)
    
    def _node(state: AgentState) -> Dict[str, Any]:
        i = state["i"]
        if i >= len(state["urls"]):
            # Finish: output final MASTERPLAN report from compressed memory
            logger.info("🎯 PHASE 4 - SYNTHESIS: Generating final MASTERPLAN REPORT")
            logger.info(f"   Compressed memory size: {len(state['memory']):,} chars")
            user = HumanMessage(content=(
                f"PHASE 4 - SYNTHESIS: Write the MASTERPLAN REPORT.\n\n"
                f"Your compressed memory contains all the key facts:\n{state['memory']}\n\n"
                f"Provide a comprehensive portfolio recommendation based on these facts."
            ))
            resp = llm.invoke([SystemMessage(content=SYSTEM), user])
            logger.info("✅ MASTERPLAN REPORT generated successfully")
            return {"messages": state["messages"] + [resp]}

        company = state["urls"][i]
        total = len(state["urls"])
        set_progress(i + 1, total, company)
        bar = progress_bar(i + 1, total)
        logger.info(f"📈 {bar} Processing: {company}")
        # Ask model to fetch report and run simulation
        user = HumanMessage(content=(
            f"Step {i+1}/{len(state['urls'])}. Company: {company}\n\n"
            f"Current compressed memory:\n{state['memory']}\n\n"
            f"1. Call fetch_annual_report for {company}\n"
            f"2. Extract KEY METRICS ONLY (Revenue, EBITDA, Pension Assets, Liability Ratio)\n"
            f"3. Then call run_monte_carlo with params '{company}_portfolio'\n"
            f"Memory will be auto-compressed after each step."
        ))
        resp = llm_with_tools.invoke([SystemMessage(content=SYSTEM)] + state["messages"] + [user])
        return {"messages": state["messages"] + [user, resp]}
    return _node


def memory_compactor_node(llm: ChatOpenAI, max_tool_chars: int = 4000):
    """
    After tools run, compress tool output into state['memory'] and discard raw output.
    """
    def _node(state: AgentState) -> Dict[str, Any]:
        # Find last tool message (if any)
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if not tool_msgs:
            return {}

        last_tool = tool_msgs[-1]
        raw = last_tool.content or ""
        clipped = raw[:max_tool_chars]  # hard bound what we feed to compactor

        logger.info(f"🗜️  COMPACTING tool output: {len(raw):,} chars → clipped to {len(clipped):,} chars")

        user = HumanMessage(content=(
            "Update the compressed memory.\n\n"
            f"Current memory:\n{state['memory']}\n\n"
            f"New tool output (clipped):\n{clipped}\n\n"
            "Write:\n"
            "1) exactly 3 bullet summary lines\n"
            "2) then an updated memory paragraph (<= 1200 chars)\n"
            "Memory must not include raw page text, only durable facts.\n"
        ))
        resp = llm.invoke([SystemMessage(content=SYSTEM), user])

        text = resp.content
        # Extract memory paragraph (simple heuristic: everything after the bullets)
        parts = text.split("\n")
        bullets = [p for p in parts if p.strip().startswith(("-", "*"))][:3]
        memory_paragraph = "\n".join(parts[len(bullets):]).strip()
        if not memory_paragraph:
            memory_paragraph = state["memory"]
        
        logger.info(f"✅ Memory updated: {len(memory_paragraph):,} chars (bounded)")

        # IMPORTANT: drop raw tool output from message history to prevent bloat
        new_messages: List[AnyMessage] = []
        for m in state["messages"]:
            if isinstance(m, ToolMessage):
                # replace with a compact tool message stub
                new_messages.append(ToolMessage(
                    content=f"[tool_output_compacted len={len(raw)} chars]",
                    tool_call_id=m.tool_call_id
                ))
            else:
                new_messages.append(m)

        return {
            "messages": new_messages + [resp],
            "memory": memory_paragraph,
            "i": state["i"] + 1,
        }
    return _node


def route_after_planner(state: AgentState) -> Literal["tools", "end"]:
    """
    If the last assistant message contains tool calls, go to tools; else end.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "end"


def main() -> None:
    logger.info("="*60)
    logger.info("🛡️  Starting MITIGATED (bounded) Pension Analysis Agent")
    logger.info("This version COMPRESSES tool outputs after each call")
    logger.info("KV cache should stay BOUNDED - no context overflow!")
    logger.info("="*60)

    llm = make_llm()

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node(llm, TOOLS))
    graph.add_node("tools", tool_node)
    graph.add_node("compact", memory_compactor_node(llm))

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner, {"tools": "tools", "end": END})
    graph.add_edge("tools", "compact")
    graph.add_edge("compact", "planner")

    app = graph.compile()

    # Same companies as naive agent - but we compress after each fetch
    companies = [
        "ACME_Corp", "GlobalTech", "SafeHaven_Insurance", "PensionFirst",
        "RetireWell", "FutureSecure", "StableGrowth", "LongHorizon"
    ]
    
    logger.info(f"📋 Processing {len(companies)} companies: {', '.join(companies)}")
    set_progress(0, len(companies))

    out = app.invoke({
        "messages": [],
        "memory": "",
        "i": 0,
        "urls": companies,  # reusing urls field for companies
    })

    logger.info("="*60)
    logger.info("🏁 FINAL OUTPUT")
    logger.info("="*60)
    print(out["messages"][-1].content)


if __name__ == "__main__":
    main()

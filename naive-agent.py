# naive_react_agent.py
"""
Simulates the "Pension Analysis" workload that causes 120B models to destabilize:
1. Ingest: Parse multiple 50+ page PDF Annual Reports (Heavy RAG)
2. Analysis: Extract financial tables and sentiment
3. Compute: Monte Carlo simulations
4. Synthesis: Aggregate into a "Masterplan" report

The failure mode: By step 4, KV Cache saturates, system swaps to disk,
inference drops to 0.1 t/s, model "forgets" earlier data (Coherence Decay).
"""
from __future__ import annotations

import logging
import warnings

# Suppress the deprecation warning for create_react_agent
warnings.filterwarnings("ignore", message="create_react_agent has been moved")

from langgraph.prebuilt import create_react_agent

from common import make_llm, TOOLS, logger, set_progress


def main() -> None:
    logger.info("="*60)
    logger.info("🚀 Starting NAIVE (unbounded) Pension Analysis Agent")
    logger.info("This version keeps ALL tool outputs in context")
    logger.info("Watch for context overflow errors!")
    logger.info("="*60)
    
    llm = make_llm()

    agent = create_react_agent(llm, TOOLS)

    # Simulates the "Pension Analysis" workflow that crashes at ~75%
    prompt = """
You are a Financial Quant Agent running a Pension Portfolio Analysis workflow.

PHASE 1 - INGEST (Heavy RAG):
Fetch annual reports for these companies: ACME_Corp, GlobalTech, SafeHaven_Insurance, 
PensionFirst, RetireWell, FutureSecure, StableGrowth, LongHorizon.

PHASE 2 - ANALYSIS:
For each report, extract and remember:
- Key financial metrics (Revenue, EBITDA, Pension Assets, Liability Ratio)
- Risk indicators and trends
- Keep a MASTER FACTS LIST that accumulates ALL findings

PHASE 3 - COMPUTE:
Run Monte Carlo simulations for each company's pension portfolio.
Parameters: "10k_runs, 30yr_horizon, inflation_adjusted"
Add all simulation results to your MASTER FACTS LIST.

PHASE 4 - SYNTHESIS (This is where 120B models typically fail):
Write a comprehensive "MASTERPLAN REPORT" that:
1. Summarizes ALL ingested data from PHASE 1
2. References ALL analysis from PHASE 2  
3. Incorporates ALL simulation results from PHASE 3
4. Provides portfolio recommendations

CRITICAL: Do NOT summarize or drop earlier data. Carry EVERYTHING forward.
The final report must reference specific numbers from each company's annual report.
"""

    logger.info("📋 Companies to process: ACME_Corp, GlobalTech, SafeHaven_Insurance, PensionFirst, RetireWell, FutureSecure, StableGrowth, LongHorizon")
    
    # Set progress for 8 companies (naive agent doesn't track per-step, but tools will show overall)
    set_progress(0, 8)
    
    try:
        result = agent.invoke({"messages": [("user", prompt)]})
        logger.info("="*60)
        logger.info("🏁 FINAL OUTPUT")
        logger.info("="*60)
        print(result["messages"][-1].content)
    except Exception as e:
        logger.error(f"❌ AGENT CRASHED: {e}")
        logger.error("This is expected - context overflow due to unbounded tool outputs!")
        raise


if __name__ == "__main__":
    main()


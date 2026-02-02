# common.py
"""
Shared utilities for agent experiments demonstrating KV cache saturation.
"""
from __future__ import annotations

import logging
import os
import random
import string

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Progress tracking
_progress = {"current": 0, "total": 0, "company": ""}


def set_progress(current: int, total: int, company: str = "") -> None:
    """Set the current progress for display."""
    _progress["current"] = current
    _progress["total"] = total
    _progress["company"] = company


def progress_bar(current: int, total: int, width: int = 30) -> str:
    """Generate a text-based progress bar."""
    if total == 0:
        return "[" + "?" * width + "]"
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    percent = 100 * current / total
    return f"[{bar}] {current}/{total} ({percent:.0f}%)"


def make_llm(model: str | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for local vLLM server."""
    return ChatOpenAI(
        model=model or os.environ.get("MODEL", "openai/gpt-oss-120b"),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url="http://localhost:8000/v1",
        temperature=0.2,
        timeout=600,
        max_retries=2,
    )


def random_financial_data(length: int = 150000) -> str:
    """Generate random 'financial report' content to defeat prefix caching."""
    sections = []
    for i in range(50):  # Simulate 50 pages
        page_content = []
        # Random company metrics
        page_content.append(f"PAGE {i+1} - Q{random.randint(1,4)} {random.randint(2020,2025)} REPORT")
        page_content.append(f"Revenue: ${random.randint(100,999)}.{random.randint(0,99)}M")
        page_content.append(f"EBITDA: ${random.randint(10,99)}.{random.randint(0,99)}M")
        page_content.append(f"Pension Fund Assets: ${random.randint(500,999)}.{random.randint(0,99)}M")
        page_content.append(f"Liability Ratio: {random.randint(70,120)}%")
        # Random narrative text
        words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 12))) 
                 for _ in range(length // 50 // 6)]
        page_content.append(' '.join(words))
        sections.append('\n'.join(page_content))
    return '\n\n---\n\n'.join(sections)


@tool
def fetch_annual_report(company: str) -> str:
    """
    Simulates downloading and parsing a 50+ page PDF Annual Report.
    This is the Heavy RAG ingestion step that fills the KV cache.
    """
    # Auto-increment progress when a new company is fetched
    if _progress["company"] != company:
        _progress["current"] += 1
        _progress["company"] = company
    
    current = _progress["current"]
    total = _progress["total"]
    bar = progress_bar(current, total) if total > 0 else ""
    logger.info(f"📄 {bar} FETCHING annual report: {company}")
    blob = f"<ANNUAL_REPORT company='{company}'>\n{random_financial_data()}\n</ANNUAL_REPORT>"
    content = f"PARSED_REPORT for {company}:\n{blob}"
    logger.info(f"✅ {bar} COMPLETED: {company} ({len(content):,} chars)")
    return content


@tool
def run_monte_carlo(params: str) -> str:
    """
    Simulates running Monte Carlo simulations for pension portfolio analysis.
    Returns large statistical output that accumulates in context.
    """
    current = _progress["current"]
    total = _progress["total"]
    bar = progress_bar(current, total) if total > 0 else ""
    logger.info(f"🎲 {bar} RUNNING Monte Carlo: {params}")
    results = []
    for run in range(100):  # Simulate 100 summary rows from 10k runs
        results.append(
            f"Run_{run:03d}: mean_return={random.uniform(-0.05, 0.15):.4f}, "
            f"var={random.uniform(0.01, 0.08):.4f}, "
            f"sharpe={random.uniform(0.5, 2.5):.3f}, "
            f"max_drawdown={random.uniform(0.05, 0.35):.3f}"
        )
    content = f"MONTE_CARLO_RESULTS ({params}):\n" + '\n'.join(results)
    logger.info(f"✅ {bar} COMPLETED Monte Carlo: {params} ({len(content):,} chars)")
    return content


@tool
def quick_math(expr: str) -> str:
    """Evaluate a simple math expression."""
    logger.info(f"🔢 EVALUATING math expression: {expr}")
    try:
        result = str(eval(expr, {"__builtins__": {}}, {}))
        logger.info(f"✅ Math result: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ Math error: {e}")
        return f"error: {e}"


# Export all tools as a list for convenience
TOOLS = [fetch_annual_report, run_monte_carlo, quick_math]

__all__ = ['make_llm', 'TOOLS', 'fetch_annual_report', 'run_monte_carlo', 'quick_math', 'logger', 'set_progress', 'progress_bar']

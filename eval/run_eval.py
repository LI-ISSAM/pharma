"""Run the 4 public evaluation questions. Usage: python -m eval.run_eval"""
import logging

from app.agent import Agent

CASES = [
    ("What are the main contraindications for warfarin?", {"dailymed-warfarin.md"}),
    ("Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?",
     {"dailymed-warfarin.md", "dailymed-apixaban.md"}),
    ("A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?",
     {"dailymed-warfarin.md", "dailymed-sertraline.md"}),
    ("A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?",
     {"dailymed-metformin.md"}),
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    agent = Agent.from_settings()
    for question, expected in CASES:
        r = agent.ask(question)
        ok = expected <= set(r.sources)
        print(f"\n{'PASS' if ok else 'FAIL'} ({r.elapsed_ms} ms, revised={r.revised}) {question[:70]}...")
        print("   sources:", r.sources, "| tools:", [c["tool"] for c in r.tool_calls])
        print("   " + r.answer[:600].replace("\n", "\n   "))
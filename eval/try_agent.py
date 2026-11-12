"""Usage: python -m eval.try_agent "your question" """
import logging
import sys

from app.agent import Agent

DEFAULT_Q = ("Compare the contraindications of warfarin and apixaban. "
             "What conditions are shared and which are unique to each drug?")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    q = " ".join(sys.argv[1:]) or DEFAULT_Q
    r = Agent.from_settings().ask(q)

    print("\nPLAN:", r.plan.sub_questions)
    print("TOOLS:", [(c["tool"], c["arguments"]) for c in r.tool_calls])
    print(f"REVIEW (available={r.review.available}, revised={r.revised}): {r.review.summary}")
    for c in r.review.claims:
        if c.verdict != "supported":
            print(f"   [{c.verdict}] {c.claim} -- {c.note}")
    for i in r.review.issues:
        print("   [issue]", i)
    print("\n=== ANSWER ===\n", r.answer)
    print("\nSOURCES:", r.sources, f"| {r.elapsed_ms} ms")
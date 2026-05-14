"""Prompts for the clean DDRG v1 main method."""

ELRG_PROMPT = """Solve the problem by constructing a reasoning graph and then give the final answer.

Rules:
1. N0 is the question.
2. Each new node must cite one or more previous parent nodes.
3. Use separate nodes for separate facts when possible.
4. When a step combines facts, cite multiple parents.
5. Output only the graph and the final answer.

Graph Structure:
N0: node content (the question)
N1 <- N0 [edge content]: node content
N2 <- N0 [edge content]: node content
N3 <- N1, N2 [edge content]: node content
...
ANS <- N3 [edge content]: node content

Final Answer: final answer
"""


ALIGNMENT_PROMPT = """Align similar steps in the sampled reasoning graphs.

Do not solve the problem. Do not choose the correct answer.
Only match nodes that talk about the same local fact or step.
Prefer nodes close to ANS. Keep only useful alignments.

Return valid JSON only:
{
  "alignments": [
    {
      "id": "A1",
      "topic": "what these nodes are about",
      "claims": [
        {"graph": 1, "node": "N2", "answer": "A", "claim": "short claim"},
        {"graph": 2, "node": "N3", "answer": "B", "claim": "short claim"}
      ]
    }
  ]
}
"""


FRONTIER_PROBE_PROMPT = """Find the smallest important disagreement.

You are given the question, support graphs, and aligned nodes.
Do not solve the full problem. Do not vote.

Pick disagreements that can change the final answer.
For each one, write one simple yes/no probe that can be answered from the
original question.

If there is no useful disagreement, return {"conflicts": []}.

Return valid JSON only:
{
  "conflicts": [
    {
      "id": "C1",
      "alignment": "A1",
      "left": {"graph": 1, "node": "N2", "answer": "A", "claim": "short claim"},
      "right": {"graph": 2, "node": "N3", "answer": "B", "claim": "short claim"},
      "issue": "short disagreement",
      "probe": {
        "question": "simple yes/no question",
        "yes_verifies": "left",
        "no_verifies": "right"
      }
    }
  ]
}
"""


PROBE_PROMPT = """Answer the probe question using only the original question.

Do not solve the full problem.
Return valid JSON only:
{
  "verdict": "YES or NO or UNKNOWN",
  "reason": "short reason"
}
"""

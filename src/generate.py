"""
generate.py
-----------
Step 5 of the RAG pipeline (optional, completes retrieval -> generation).

Takes a user question, retrieves top_k chunks via retrieve.py, assembles
a grounded prompt, and sends it to Claude via the Anthropic API to
produce a final, cited answer.

Requires an ANTHROPIC_API_KEY environment variable.

Run:
    python src/generate.py "What is the surgical timing for pilon fractures?"
"""

import os
import sys

import anthropic

from retrieve import Retriever, build_prompt, DEFAULT_TOP_K

MODEL = "claude-sonnet-4-6"  # swap for whichever Claude model your API access includes


def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    retriever = Retriever()
    hits = retriever.retrieve(query, top_k=top_k)
    prompt = build_prompt(query, hits)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    query = " ".join(sys.argv[1:]) or "What is the surgical timing for pilon fractures?"
    print(answer_question(query))


if __name__ == "__main__":
    main()

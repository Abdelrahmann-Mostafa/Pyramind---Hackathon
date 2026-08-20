# 🎬 Hypertension RAG + Agents Demo Script (5 min)

## Opening (30 sec)

"We built a clinical decision support system for hypertension management using RAG + agents.

**The Problem:** Doctors need fast, evidence-based recommendations. Manual guideline lookup is slow.

**Our Solution:** RAG retrieves from ESC 2021 guidelines + agents personalize based on patient profile."

---

## Demo 1: Basic RAG (1 min)

**Open Streamlit app → disable "Use Agent Reasoning"**

Type query: "What is the recommended blood pressure target?"

[System shows response with citations]

Point out:
- ✅ Answer is grounded in ESC 2021 § 3.2
- ✅ Shows exact page number and section
- ✅ No hallucination (can't make up info)

---

## Demo 2: Agent-Personalized (2 min)

**Enable "Use Agent Reasoning"**

Type: "I'm 62 with diabetes. What should my blood pressure be?"

[Agent asks follow-up questions]

Agent: "I see you have diabetes. Let me search the guidelines specifically for your situation..."

[System retrieves diabetes-specific target: <130/80]

Point out:
- ✅ Agent asked clarifying question (added diabetes context)
- ✅ Different recommendation than non-diabetic patients
- ✅ Multi-hop reasoning: age + comorbidity → personalized target
- ✅ Still grounded in ESC 2021

---

## Demo 3: Refusal (45 sec)

Type: "What are the side effects of aspirin?"

[System refuses: "outside scope"]

Point out:
- ✅ Safety mechanism prevents out-of-scope answers
- ✅ Won't hallucinate when guidelines don't cover topic
- ✅ Explicit refusal better than wrong answer

---

## Closing (1 min)

"**Architecture:**
1. Ingestion: Parse ESC 2021 PDF → chunked + indexed in Chroma
2. Retrieval: Semantic search with embeddings
3. Agent: Claude reasons about which section applies
4. Generation: Grounded in official guideline text

**Why Agents Matter:**
- Without agents: returns generic BP target (140/90)
- With agents: personalizes (130/80 for diabetes)
- Same retrieval, better results through reasoning

**Future:**
- Add drug interaction checking
- Integrate patient EHR data
- Multi-guideline cross-reference (NICE, ACC/AHA)
"

---

## Q&A Prep

**Q: What if the guideline doesn't cover something?**
A: Agent recognizes insufficient context and refuses - safety first.

**Q: Can you add more guidelines?**
A: Yes, ingest any official PDF. Same architecture scales.

**Q: Is this production-ready?**
A: Good for clinical decision support, not autonomous treatment decisions.

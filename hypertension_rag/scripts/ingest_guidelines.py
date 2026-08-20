#!/usr/bin/env python3
"""
Ingest ESC 2021 Hypertension Guideline into Chroma
"""

import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def ingest_mock_guideline():
    """Ingest the mock ESC guideline."""
    
    # Load embedding model
    embedding_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    print("[*] Embedding model loaded")
    
    # Initialize Chroma
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection(name="hypertension-guidelines")
    print("[*] Chroma collection ready")
    
    # Mock guideline sections
    sections = [
        {
            "id": "esc_2021_1.0",
            "section": "1.0",
            "title": "Introduction",
            "page": 5,
            "content": "Blood pressure management is critical in diabetes management. This guideline provides recommendations for screening, diagnosis, and treatment of hypertension based on current evidence."
        },
        {
            "id": "esc_2021_2.1",
            "section": "2.1",
            "title": "Blood Pressure Measurement",
            "page": 8,
            "content": "Office blood pressure should be measured using validated automatic devices. Blood pressure should be measured in sitting position after 5 minutes of rest. Measurements should be repeated every 1-2 minutes and the average recorded."
        },
        {
            "id": "esc_2021_3.2",
            "section": "3.2",
            "title": "Target Blood Pressure by Risk Category",
            "page": 15,
            "content": "For patients with diabetes, ESC 2021 recommends a systolic blood pressure target of less than or equal to 130 mmHg and diastolic target of less than or equal to 80 mmHg. This more stringent target is recommended for high-risk patients with diabetes and established cardiovascular disease."
        },
        {
            "id": "esc_2021_4.2.1",
            "section": "4.2.1",
            "title": "Initial Antihypertensive Therapy",
            "page": 22,
            "content": "ACE inhibitors or angiotensin receptor blockers (ARBs) are recommended as first-line antihypertensive agents in patients with diabetes. Both classes offer cardiovascular and renal protection. Calcium channel blockers are alternative first-line agents. Beta-blockers should be used cautiously in diabetes due to potential metabolic side effects including worsening of glucose control."
        },
        {
            "id": "esc_2021_4.3.1",
            "section": "4.3.1",
            "title": "Elderly Patients",
            "page": 28,
            "content": "For elderly patients aged 65 years or older with hypertension, a less stringent blood pressure target of less than 140/90 mmHg is recommended. Treatment should be initiated cautiously with regular monitoring for hypotension and adverse effects. Dose titration should occur slowly."
        },
    ]
    
    # Add sections to Chroma
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    for section in sections:
        ids.append(section["id"])
        documents.append(section["content"])
        metadatas.append({
            "section_number": section["section"],
            "section_title": section["title"],
            "page_number": section["page"],
            "document_name": "ESC 2021 Hypertension Guidelines",
        })
        embeddings.append(embedding_model.encode(section["content"]).tolist())
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    
    print(f"[✅] Ingested {len(sections)} guideline sections")
    return collection

if __name__ == "__main__":
    ingest_mock_guideline()
    print("✅ Guideline ingestion complete")

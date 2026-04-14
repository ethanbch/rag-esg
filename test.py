import os

import requests

key = os.getenv("ALBERT_API_KEY")
headers = {"Authorization": f"Bearer {key}"}
base = "https://albert.api.etalab.gouv.fr/v1"
FILE_PATH = "/Users/ethan.bcht/Dev/esg/totalenergies_sustainability-climate-2024-progress-report_2024_en_pdf.pdf"

QUESTIONS = [
    "Scope 1 and Scope 2 GHG emissions for the latest reporting year",
    "Climate targets 2030 science based",
    "ESRS E1 climate change transition plan",
    "EU Taxonomy CapEx revenue aligned percentage",
    "Governance board committees sustainability reporting",
    "Climate risks and opportunities",
    "Workforce health safety diversity ESRS S1",
    "Biodiversity water impacts ESRS E2 E3 E4",
    "Value chain due diligence suppliers human rights",
    "Assurance level limited reasonable sustainability",
]

# 1. Créer collection + uploader
col_id = requests.post(
    f"{base}/collections",
    headers=headers,
    json={"name": "Total_CSRD_chat", "model": "BAAI/bge-m3"},
).json()["id"]
print(f"Collection : {col_id}")

with open(FILE_PATH, "rb") as f:
    doc_id = requests.post(
        f"{base}/documents",
        headers=headers,
        data={"collection_id": col_id},
        files={"file": (os.path.basename(FILE_PATH), f, "application/pdf")},
    ).json()["id"]
print(f"Document : {doc_id}\n")

# 1.5 Exporter le document "parsé" (chunks)
print("--- Export du document (Chunks) ---")
all_text = []
offset = 0
limit = 100
while True:
    res = requests.get(
        f"{base}/documents/{doc_id}/chunks",
        headers=headers,
        params={"limit": limit, "offset": offset},
    ).json()
    chunks = res.get("data", [])
    if not chunks:
        break
    for c in chunks:
        if "content" in c:
            all_text.append(c["content"])
    offset += limit

parsed_text = "\n\n".join(all_text)
output_path = FILE_PATH.replace(".pdf", "_parsed.txt")
with open(output_path, "w", encoding="utf-8") as out:
    out.write(parsed_text)
print(f"✅ Parsing exporté (via chunks) : {output_path} ({len(parsed_text):,} chars)\n")

# 2. Récupérer les chunks pour toutes les questions
context = ""
for question in QUESTIONS:
    results = requests.post(
        f"{base}/search",
        headers=headers,
        json={"prompt": question, "collections": [col_id], "k": 3},
    ).json()
    for r in results.get("data", []):
        text = r["chunk"]["content"]
        context += f"\n---\n{text}"

print(f"Contexte constitué : {len(context)} chars\n")

# 3. Chat avec le contexte injecté
response = requests.post(
    f"{base}/chat/completions",
    headers=headers,
    json={
        "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "messages": [
            {
                "role": "system",
                "content": f"Answer strictly based on these document extracts:\n{context}",
            },
            {
                "role": "user",
                "content": """Answer each question based on the extracts above:
1. Scope 1 and Scope 2 GHG emissions (latest year)?
2. Climate targets 2030 — science-based?
3. ESRS E1 and transition plan?
4. EU Taxonomy CapEx and revenue alignment?
5. Governance arrangements?
6. Main climate risks and opportunities?
7. Workforce health, safety, diversity (ESRS S1)?
8. Biodiversity and water (ESRS E2–E4)?
9. Value chain due diligence?
10. Assurance level?""",
            },
        ],
        "temperature": 0.1,
    },
).json()

answer = response["choices"][0]["message"]["content"]

# 4. Export de la réponse CSRD
answer_path = FILE_PATH.replace(".pdf", "_csrd_analysis.txt")
with open(answer_path, "w", encoding="utf-8") as out:
    out.write(answer)
print(f"✅ Analyse CSRD exportée : {answer_path}\n")

print("=== RÉPONSE CSRD ===")
print(answer)

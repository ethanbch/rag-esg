import pymupdf4llm
chunks = pymupdf4llm.to_markdown("downloads/totalenergies_sustainability-climate-2025-progress-report_2025_en.pdf", page_chunks=True)
print(chunks[0].keys())
print(chunks[1]["metadata"])

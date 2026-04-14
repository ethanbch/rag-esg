import glob
import os

from config import ALBERT_API_KEY, CHUNKING_STRATEGY
from ingestion_pipeline.step01_parsing import extract_text_from_pdf
from ingestion_pipeline.step02_chunking import chunk_pdf_text
from ingestion_pipeline.step03_indexing import index_chunks


def process_pdf(pdf_path, collection_name):
    print(f"Processing {pdf_path}...")

    if not ALBERT_API_KEY:
        print("Error: ALBERT_API_KEY environment variable is not set.")
        return

    # 1. Extract text from PDF.
    print("  Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text.strip():
        print(f"  No text extracted for {pdf_path}.")
        return

    # 2. Chunk text with the configured chunking strategy.
    print("  Chunking extracted text...")
    all_chunks = chunk_pdf_text(
        pdf_path,
        extracted_text,
        strategy=CHUNKING_STRATEGY,
    )

    if not all_chunks:
        print(f"  No chunks returned for {pdf_path}.")
        return

    # 3. Embed chunks with Albert BGE-M3 and store in ChromaDB.
    print(
        f"  Storing {len(all_chunks)} chunks in local ChromaDB collection '{collection_name}'..."
    )
    index_chunks(collection_name, all_chunks)

    print(f"Finished storing chunks for {pdf_path} in ChromaDB.\n")


def main():
    download_dir = "downloads"

    if not os.path.exists(download_dir):
        print(
            f"Directory {download_dir} not found. Please run step00_scraping.py first."
        )
        return

    pdf_files = glob.glob(os.path.join(download_dir, "*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {download_dir}.")
        return

    print(f"Found {len(pdf_files)} PDFs to process.")

    for pdf_path in pdf_files:
        # Create a collection name based on the filename (alphanumeric and underscores only)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        clean_name = "".join([c if c.isalnum() else "_" for c in base_name])

        process_pdf(pdf_path, clean_name)


if __name__ == "__main__":
    main()

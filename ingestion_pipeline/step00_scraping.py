import os
from urllib.parse import urlparse

import yaml
from curl_cffi import requests


def download_pdf(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, impersonate="chrome110", stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded: {output_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def main():
    yaml_file = "company_list.yaml"

    # Load YAML file
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    companies = data.get("TARGET_COMPANIES", [])

    # Ensure a directory exists for downloads, or just save them in the current directory
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)

    for company in companies:
        name = company.get("name")
        pdf_url = company.get("pdf_url")

        if name and pdf_url:
            # Extract basic filename from URL or generate one
            parsed_url = urlparse(pdf_url)
            filename = os.path.basename(parsed_url.path)

            # If the URL doesn't have a clear filename, create one from the company name
            if not filename or not filename.endswith(".pdf"):
                safe_name = name.lower().replace(" ", "_").replace("'", "")
                filename = f"{safe_name}_report.pdf"

            output_path = os.path.join(output_dir, filename)
            download_pdf(pdf_url, output_path)


if __name__ == "__main__":
    main()

import os
import glob
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pdf2image import convert_from_path
from io import BytesIO
import openai
import backoff
from openai import AzureOpenAI

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)
client = AzureOpenAI(azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                     api_key=os.getenv('AZURE_OPENAI_KEY'),
                     api_version=os.getenv('AZURE_OPENAI_VERSION'))

def convert_pdf_to_base64_image(pdf_path):
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    img_buffer = BytesIO()
    images[0].save(img_buffer, format="PNG")
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")


@backoff.on_exception(backoff.expo, (openai.RateLimitError), max_tries=5)
def analyze_document(base64_image):
    response = client.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert the image to an HTML table. The output should begin with <table> and end with </table>. Specify rowspan and colspan attributes when they are greater than 1. Do not specify any other attributes. Only use table related HTML tags, no additional formatting is required.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content


def process_pdf(pdf_path: str):
    output_path = pdf_path.replace("pdfs", "gpt4o").replace(".pdf", ".json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        base64_image = convert_pdf_to_base64_image(pdf_path)
        result = analyze_document(base64_image)

        with open(output_path, "w") as f:
            json.dump({"html_table": result}, f, indent=2)

        print(output_path)

        return pdf_path, None
    except openai.RateLimitError:
        return pdf_path, "Rate limit"
    except Exception as e:
        return pdf_path, str(e)


def process_all_pdfs(pdfs: list[str]):
    max_workers = 5  # Adjust based on your API rate limits

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdfs}

        progress_bar = tqdm(total=len(pdfs), desc="Processing PDFs")

        for future in as_completed(futures):
            pdf_path, error = future.result()
            if error:
                print(f"Error processing {pdf_path}: {error}")
            progress_bar.update(1)

        progress_bar.close()

    print(f"Processed {len(pdfs)} PDFs")


if __name__ == "__main__":
    process_all_pdfs(pdfs)

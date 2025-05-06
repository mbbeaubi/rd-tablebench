import os
import glob
from tqdm import tqdm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from marker.converters.table import TableConverter
from marker.config.crawler import crawler

base_path = os.path.expanduser("~/data/human_table_benchmark")
pdfs = glob.glob(os.path.join(base_path, "**", "*.pdf"), recursive=True)

config = {
    "output_format": "html",
    "disable_tqdm": False,
    "force_ocr": True,
    'enable_table_ocr': True
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

def process_all_pdfs(pdfs: list[str]):
    for pdf in tqdm(pdfs[:10]):
        output_path = pdf.replace("pdfs", "marker").replace(".pdf", ".html")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rendered = converter(pdf)
        text, type, images = text_from_rendered(rendered)
        print(type)
        with open(output_path, 'w', encoding='UTF-8') as fp:
            fp.write(text)


if __name__ == "__main__":
    print(",".join(crawler.attr_set))
    process_all_pdfs(pdfs)

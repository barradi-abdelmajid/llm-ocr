from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import torch
print("CUDA Available:", torch.cuda.is_available())

source = "pdf_idk.pdf" 

converter= DocumentConverter()

result=converter.convert(source)

markdown_output = result.document.export_to_markdown()

with open("document_output.md", "w", encoding="utf-8") as file:
    file.write(markdown_output)

print("Conversion complete! Your file has been saved as 'document_output.md'")
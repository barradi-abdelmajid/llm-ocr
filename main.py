from docling.document_converter import DocumentConverter


source = "D:\documents\github\Llm-ocr\llm ocr\pdf_idk.pdf" 

converter = DocumentConverter()

result = converter.convert(source)

markdown_output = result.document.export_to_markdown()

with open("document_output.md", "w", encoding="utf-8") as file:
    file.write(markdown_output)

print("Conversion complete! Your file has been saved as 'document_output.md'")
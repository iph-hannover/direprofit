# from pdf2image import convert_from_path
# import pytesseract
import os
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from docling.document_converter import DocumentConverter

# PDF in Bilder umwandeln
dir_list = os.listdir('dokumente')
dir_Used = os.listdir('usedQuellen')
for nr in reversed(range(len(dir_list))):
    for nrQuelle in range(len(dir_Used)):
        if dir_list[nr][:-4] in dir_Used[nrQuelle]:
            del dir_list[nr]
            test = 7
            break
quellenOrdner = "dokumente"
for quellenItem in dir_list:
    quellenPath = os.path.join(quellenOrdner, quellenItem)
    converter = DocumentConverter()
    result = converter.convert(quellenPath)
    clearText = result.document.export_to_markdown()
    test = 7
    with open(os.path.join('usedQuellen',quellenItem[:-4] +'.txt'),'w', encoding="utf-8") as txt_file:
        txt_file.write(clearText)
    test = 7
print("Alle Dateien in Dokumente sind umgewandelt.")
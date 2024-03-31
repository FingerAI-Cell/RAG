import argparse 
import numpy as np 
import os 
import json
from llama_index import SimpleDirectoryReader
from src.pdf_processor import PDFProcessor

def main(cli_argse): 
    with open(cli_argse.config_file, "r", encoding="utf-8") as f:
        pdf_config = json.load(f)
    
    file_path = os.path.join(pdf_config['data_path'], pdf_config['file_name'])
    # print(f'file_path: {file_path}')

    pdf_processor = PDFProcessor(pdf_config)
    pdf_processor.set_num_mapper()
    pdf_file = SimpleDirectoryReader(input_files=[file_path]).load_data()
    # print(pdf_file)

    for idx in range(len(pdf_file)):
        pdf_file[idx] = pdf_processor.text_cleanse(pdf_file[idx])
    
    start_doc_no = pdf_processor.get_start_point(pdf_file)
    print(start_doc_no)
    pdf_processor.create_new_documents(pdf_file)
    new_docs = pdf_processor.new_doc
    print(len(new_docs), np.shape(new_docs))
    # print(new_docs[0][0], new_docs[0][1])
    for idx in range(len(new_docs)):
        print(new_docs[idx].metadata)
        print(new_docs[idx].text, end='\n\n')

if __name__ == '__main__':
    global cli_argse

    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_file", type=str, default='/rag/config/pdf_config.json')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)

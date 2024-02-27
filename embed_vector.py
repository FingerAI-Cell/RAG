# -*- coding: cp949 -*- 
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from src.vector_store import LocalStorage, ChromaDB
import argparse 
import pandas as pd 
import os
import json

def main(cli_argse):
    ''' 
    Vector DB�� Document�� ������ �ε����� ��ȯ�ؼ� �����ϴ� �ڵ�
    '''
    with open(os.path.join(cli_argse.config_path, 'vectordb_config.json'), "r", encoding="utf-8") as f:
        db_config = json.load(f)

    chromadb = ChromaDB(db_config)
    chromadb.connect(cli_argse.db_type)

    loop_flag = True 
    while loop_flag: 
        collection_list = list(map(lambda x: x.name, chromadb.client.list_collections()))
        print(f'Vector DB collections: {collection_list}')
        print(f'-' * 10)
        db_collection = chromadb.get_collection(db_config['collection_name'])
        print(f"���� collection: {db_config['collection_name']}")
        print(f"collection {db_config['collection_name']} ����: {db_collection.count()}��")

        act_type = 0; act_flag = 0 
        if db_config['collection_name'] not in collection_list:
            print(f"���� ������ �۾�: create")
            act_flag = input('��� �Ͻðڽ��ϱ� (0: yes, 1: no)?: ')
            if act_flag == '0':
                print(f'Collection�� �����մϴ�.')
                act_type = '0' 
            else:
                loop_flag = False 
                break  
        else: 
            print(f"���� ������ �۾�: insert, delete")
            act_flag = input('��� �Ͻðڽ��ϱ� (0: yes, 1: no)?: ')
            if act_flag == '0': 
                act_type = input('���� �۾��� �Ͻó��� (1: insert, 2: delete)?: ')
            else:
                loop_flag = False 
                break
        
        if act_type == '0' or act_type == '1':
            db_store = ChromaVectorStore(chroma_collection=db_collection)
            db_storage = StorageContext.from_defaults(vector_store=db_store)
            
            data = pd.read_csv(os.path.join(db_config['data_path'], db_config['file_name']))
            nona = data.copy()
            nona.dropna(subset=db_config['emb_col'], inplace=True)   # �Ӻ����� ������ NULL �� ���� 
            
            metadatas = dict() 
            for value in db_config['metadata_col'].values():   # 
                metadatas[value] = nona[value].values.tolist()
            datas = nona[db_config['emb_col']].values.tolist()
            documents = chromadb.get_documents(datas, metadatas)
        
        if act_type == '0':
            print(f"Collection {db_config['collection_name']}�� ���� Index�� �����մϴ�.")
            print(f"���ϸ�: {db_config['file_name']}")
            index = chromadb.create_index(db_storage, documents)
            index.set_index_id(f"scrapping_{db_config['collection_name']}_chroma")
        elif act_type == '1':
            print(f"Collection {db_config['collection_name']} �ε����� Document�� �߰��մϴ�.")
            index = chromadb.get_vector_index(db_store)
            index = chromadb.insert_index(index, documents)
            index.set_index_id(f"scrapping_{db_config['collection_name']}_chroma")
            print(f'Index {index.index_id}')
        elif act_type == '2':
            print(f"Collection {db_config['collection_name']}�� �����մϴ�.")
            chromadb.delete_collection(db_config['collection_name']) 
        else:
            print(f'0 �Ǵ� 1 �Ǵ� 2�� �Է����ּ���.')
            continue 

        flag = input('�����Ͻðڽ��ϱ� ? (0: ����, 1: ���): ')
        if flag == '0':
            loop_flag = False 
        elif flag == '1':
            loop_flag = True 

if __name__ == '__main__':
    global cli_argse

    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_path", type=str, default='/rag/config')
    cli_parser.add_argument("--db_type", type=int, default=0, help="0: local storage, 1: network (docker)")
    cli_argse = cli_parser.parse_args()
    main(cli_argse)
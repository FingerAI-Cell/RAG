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
    Vector DB에 Document의 내용을 인덱스로 변환해서 저장하는 코드
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
        print(f"현재 collection: {db_config['collection_name']}")
        print(f"collection {db_config['collection_name']} 개수: {db_collection.count()}개")

        act_type = 0; act_flag = 0 
        if db_config['collection_name'] not in collection_list:
            print(f"수행 가능한 작업: create")
            act_flag = input('계속 하시겠습니까 (0: yes, 1: no)?: ')
            if act_flag == '0':
                print(f'Collection을 생성합니다.')
                act_type = '0' 
            else:
                loop_flag = False 
                break  
        else: 
            print(f"수행 가능한 작업: insert, delete")
            act_flag = input('계속 하시겠습니까 (0: yes, 1: no)?: ')
            if act_flag == '0': 
                act_type = input('무슨 작업을 하시나요 (1: insert, 2: delete)?: ')
            else:
                loop_flag = False 
                break
        
        if act_type == '0' or act_type == '1':
            db_store = ChromaVectorStore(chroma_collection=db_collection)
            db_storage = StorageContext.from_defaults(vector_store=db_store)
            
            data = pd.read_csv(os.path.join(db_config['data_path'], db_config['file_name']))
            nona = data.copy()
            nona.dropna(subset=db_config['emb_col'], inplace=True)   # 임베딩할 데이터 NULL 값 제거 
            
            metadatas = dict() 
            for value in db_config['metadata_col'].values():   # 
                metadatas[value] = nona[value].values.tolist()
            datas = nona[db_config['emb_col']].values.tolist()
            documents = chromadb.get_documents(datas, metadatas)
        
        if act_type == '0':
            print(f"Collection {db_config['collection_name']}에 대한 Index를 생성합니다.")
            print(f"파일명: {db_config['file_name']}")
            index = chromadb.create_index(db_storage, documents)
            index.set_index_id(f"scrapping_{db_config['collection_name']}_chroma")
        elif act_type == '1':
            print(f"Collection {db_config['collection_name']} 인덱스에 Document를 추가합니다.")
            index = chromadb.get_vector_index(db_store)
            index = chromadb.insert_index(index, documents)
            index.set_index_id(f"scrapping_{db_config['collection_name']}_chroma")
            print(f'Index {index.index_id}')
        elif act_type == '2':
            print(f"Collection {db_config['collection_name']}을 삭제합니다.")
            chromadb.delete_collection(db_config['collection_name']) 
        else:
            print(f'0 또는 1 또는 2를 입력해주세요.')
            continue 

        flag = input('종료하시겠습니까 ? (0: 종료, 1: 계속): ')
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
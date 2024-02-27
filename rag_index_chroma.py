# -*- coding: cp949 -*- 
import json 
import os 
import argparse 
import time 
from llama_index import QueryBundle 
from src.vector_store import ChromaDB
from src.retriever import IndexRetriever 
from src.generator import ResponseGenerator 

def main(cli_argse):
    '''
    [Setting] 
    vectordb: chroma, collection: desc, qualification, features
    emb model: kakaobank/kf-deberta-base
    generation model: davidkim205/komt-mistral-7b-v1
    '''
    with open(os.path.join(cli_argse.config_path, 'vectordb_config.json'), "r", encoding="utf-8") as f:
        db_config = json.load(f)

    with open(os.path.join(cli_argse.config_path, 'config.json'), "r", encoding="utf-8") as f:
        config = json.load(f)

    vectordb = ChromaDB(db_config)
    vectordb.connect(cli_argse.db_type)
    chroma_collection = vectordb.get_collection()
    vector_store = vectordb.get_vector_store(chroma_collection)
    vector_idx = vectordb.get_vector_index(vector_store)
    
    index_retriever = IndexRetriever(vector_idx, vector_store)
    response_generator = ResponseGenerator(config)
    response_generator.set_gpu()
    response_generator.set_generation_config()

    '''
    Inference
    '''
    print("-" * 10)
    print("RAG Architecture Chatbot Model")
    print("-" * 10, end='\n\n')
    flag = True 

    user_name = input('당신의 이름은 ?')
    while flag:
        query = input(f"{user_name}: ")
        query_bundle = QueryBundle(query)
        # print(query_bundle)
        # start = time.time() 
        retrieved_nodes = index_retriever.retrieve_nodes(query_bundle)
        # print(retrieved_nodes)
        # print(f'관련 노드 추출에 걸린 시간: {time.time() - start}')

        processed_nodes = index_retriever.postprocess_node(query_bundle, retrieved_nodes)
        # print(f'노드 후처리에 걸린 시간: {time.time() - start}')
        node_info = index_retriever.get_node_info(processed_nodes[0])

        # Augment 
        context = f"'{node_info['name']}'은 {node_info['text']} 관련 상품입니다."
        print(f'관련 정보: {context}')
        response_generator.set_prompt_template(query, context)
        response = response_generator.generate_response()

        print(f"Chatbot: {response}")
        flag = input("계속 하시겠습니까 ? (Y / N)")
        if flag in ['Y', 'y', '네']:
            flag = True 
        elif flag in ['N', 'n', '아니오']: 
            flag = False 
            print("대화를 종료합니다.")

if __name__ == '__main__':
    global cli_argse

    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_path", type=str, default='/rag/config')
    cli_parser.add_argument("--db_type", type=int, default=0, help="0: local storage, 1: network (docker)")
    cli_argse = cli_parser.parse_args()
    main(cli_argse)
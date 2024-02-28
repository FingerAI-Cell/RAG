import argparse 
import json 
import os
from llama_index import QueryBundle 
from llama_index.tools import RetrieverTool
from src.vector_store import ChromaDB
from src.retriever import IndexRetriever 
from src.generator import ResponseGenerator 

def main(cli_argse):
    '''
    [Setting] 
    vectordb: chroma, collection: card, loan, deposit
    emb model: kakaobank/kf-deberta-base
    generation model: davidkim205/komt-mistral-7b-v1
    '''
    with open(os.path.join(cli_argse.config_path, 'vectordb_config.json'), "r", encoding="utf-8") as f:
        db_config = json.load(f)

    with open(os.path.join(cli_argse.config_path, 'config.json'), "r", encoding="utf-8") as f:
        config = json.load(f)

    vectordb = ChromaDB(db_config)
    vectordb.connect(cli_argse.db_type)
    collection_list = list(map(lambda x: x.name, vectordb.client.list_collections()))
    collection_map = {"card": "카드", "loan": "대출", "deposit": "예금"}
    
    vector_collection = dict()
    vector_store = dict()
    vector_idx = dict()
    vector_retriever = dict()
    vector_tool = dict()   
    
    for collection in collection_list:
        vector_collection[collection] = vectordb.get_collection(collection)
        vector_store[collection] = vectordb.get_vector_store(vector_collection[collection])
        vector_idx[collection] = vectordb.get_vector_index(vector_store[collection])
        vector_retriever[collection] = vector_idx[collection].as_retriever(similarity_top_k=5)
        vector_tool[collection] = RetrieverTool.from_defaults(retriever=vector_retriever[collection], \
                                    description=(f"{collection_map[collection]} 상품에 대해 물어볼 때 대답해줘"))

    router_retriever = IndexRetriever(embed_model=vectordb.emb_model, vector_tool=vector_tool,\
                                embedding_service=vectordb.embedding_service, retrieve_type='router')
    
    # print(f"test: {router_retriever.retrieve('청년들')}")
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

    user_name = input('당신의 이름은 ?: ')
    while flag:
        query = input(f"{user_name}: ")
        query_bundle = QueryBundle(query)
        # print(query_bundle)
        # start = time.time() 
        retrieved_nodes = router_retriever.retrieve_nodes(query_bundle)
        print(len(retrieved_nodes))
        print(router_retriever.get_node_info(retrieved_nodes[0]))
        # print(router_retriever.get_node_info(retrieved_nodes[1]))
        # print(router_retriever.get_node_info(retrieved_nodes[2]))
        # print(router_retriever.get_node_info(retrieved_nodes[3]))
        # print(retrieved_nodes)
        # print(f'관련 노드 추출에 걸린 시간: {time.time() - start}')
        # print(retrieved_nodes)
        processed_nodes = router_retriever.postprocess_node(query_bundle, retrieved_nodes)
        # print(f'노드 후처리에 걸린 시간: {time.time() - start}')
        node_info = router_retriever.get_node_info(processed_nodes[0])

        # Augment 
        context = f"'{node_info['name']}'은 {node_info['text']} 관련 상품입니다."
        # print(f'관련 정보: {context}')+
        response_generator.set_prompt_template(query, context)
        response = response_generator.generate_response()

        print(f"Chatbot: {response}")
        flag = input("계속 하시겠습니까 ? (Y / N): ")
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

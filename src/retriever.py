from llama_index.retrievers import VectorIndexRetriever 
from llama_index.postprocessor import SimilarityPostprocessor 
from llama_index.postprocessor import SentenceTransformerRerank 
from llama_index.postprocessor import SimilarityPostprocessor, CohereRerank
from llama_index.selectors import EmbeddingSingleSelector
from llama_index.retrievers import RouterRetriever

class IndexRetriever():
    def __init__(self, vector_idx=None, embed_model=None, embedding_service=None, top_k=None, vector_tool=None, retrieve_type=None, verbose=True):
        if retrieve_type == 'router':
            self.retriever = RouterRetriever(
                selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model),
                retriever_tools=list(vector_tool.values()),
                service_context = embedding_service
            )
        else:
            top_k=5 
            self.retriever = VectorIndexRetriever(index=vector_idx, service_context=embedding_service, similarity_top_k=top_k, verbose=verbose)

    def retrieve_nodes(self, query):
        nodes = self.retriever.retrieve(query)
        return nodes 

    def get_node_info(self, retrieved_node):
        '''
        id, text, score 
        '''
        node_info = dict()
        node_info['id'] = retrieved_node.node.relationships['1'].node_id
        node_info['name'] = retrieved_node.node.relationships['1'].metadata['name']
        node_info['text'] = retrieved_node.node.text 
        node_info['score'] = retrieved_node.score 
        return node_info 

    def postprocess_node(self, query_bundle, nodes, sim_cutoff=0.3, reranker_top_n=1, with_reranker=False):
        node_postprocessors = SimilarityPostprocessor(similarity_cutoff=sim_cutoff)
        processed_nodes = node_postprocessors.postprocess_nodes(nodes)
        
        if with_reranker:   # 나중에 Cohere 모델로 변경 후 속도 측정 
            reranker = SentenceTransformerRerank(
                model='bongsoo/albert-small-kor-cross-encoder-v1',
                top_n=reranker_top_n,
            )
            reranked_nodes = reranker.postprocess_nodes(
                processed_nodes, query_bundle
            )
            return reranked_nodes
        return processed_nodes 
    
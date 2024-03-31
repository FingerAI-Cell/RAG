import re 
import numpy as np 
from llama_index import Document
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding

class PDFProcessor():
    def __init__(self, config):
        self.file_name = config['file_name'].split('.')[0]
        self.spot_point = ''   # '제 1 장', '제 2 장', ... , '부칙' 등과 같은 정보 저장 
        self.doc_no = 0
        self.new_doc = []
    
    def set_num_mapper(self):
        self.num_mapper = dict({'①':'제1항', '②':'제2항', '③':'제3항', '④':'제4항', '⑤':'제5항', '⑥':'제6항',\
                   '⑦':'제7항', '⑧': '제8항', '⑨': '제9항', '⑩': '제10항', '⑪': '제11항', '⑫': '제12항',\
                  '⑬':'제13항', '⑭':'제14항', '⑮':'제15항', '⑯':'제16항', '⑰':'제17항', '⑱':'제18항', '⑲':'제19항', '⑳':'제20항'})

    def text_cleanse(self, document):
        '''
        여러 공백 문자열 단일 공백 문자로 변환 
        여러 줄변환 문자 줄변환 문자로 변환 (\n x 2~ => \n x 2) 
        문서 맨 윗 내용이 페이지 번호인 경우 페이지 번호 제거 
        '''
        document.text = re.sub('[^A-Za-z0-9\'\"\-가-힣(){}\\n[]]', '', document.text) 
        for num in self.num_mapper.keys():
            document.text = document.text.replace(num, self.num_mapper[num])
        document.text = re.sub(' +', ' ', document.text)
        document.text = document.text.strip()
        text_list = document.text.splitlines(True)
        
        # pdf 문서 처음이 페이지 번호로 시작하는 경우 해당 내용 제거 
        if text_list[0].startswith('페이지') or text_list[0].startswith(document.metadata['page_label']):
            text_list = text_list[1:]

        new_text = []
        for line in text_list:
            if line == '\n' or line == ' \n':
                continue
            new_text.append(line)
        document.text = ''.join(new_text)
        return document
    
    def get_start_point(self, documents):
        '''
        목차 등을 걸러내고 문서 본문이 시작되는 지점을 찾는 함수
        '''
        s_point = 1
        for doc in documents:
            if len(re.findall(r'제 *1 *장', doc.text)) != 0 and (len(re.findall(r'목 *차', doc.text)) == 0 and len(re.findall(r'차 *례', doc.text)) == 0): 
                s_point = doc.metadata['page_label']
                break
        self.s_point = int(s_point) - 1
        return int(s_point) - 1 

    def get_split_spots(self, document):
        '''
        문서를 분리하는 지점을 찾아내는 함수 
            Step 1. 문서를 줄 단위로 분리 
            Step 2. 분리된 라인에서 '제 1 장, 제 2 장 ~, 부칙 같은 표현이 들어있는 지점을 찾아냄
            Step 3. 찾은 spot들을 split_spots dictionary에 저장   (key: split line number, value: '제1장', '제2장', ... ,'부칙')
            return split_spots 
        ''' 
        text_list = document.text.splitlines(True)
        split_spots = dict()
        for idx, text in enumerate(text_list):
            content_spots = re.findall(r'제 *[0-9] *장', text)
            extra_spots = re.findall(r'부 *칙', text)
            
            if len(content_spots) == 0 and extra_spots == 0:   # 해당 텍스트가 본문이면 
                continue
            elif len(content_spots) > 0:
                split_spots[idx] = content_spots
            elif len(extra_spots) > 0:
                split_spots[idx] = extra_spots
        return split_spots
    
    def create_doc(self, ):
        pass
    
    def split_doc(self, split_spots: dict, document):
        '''
        주어진 split spots에 따라 문서를 분리하는 함수 
        '''
        text_list = document.text.splitlines(True)
        spot_list = list(split_spots.keys())    # line 1, line 2, ...  
        # splitted_doc = []

        for idx in range(len(spot_list)):
            if idx == 0 and len(spot_list) != 1:   # spot_list가 한 개 이상 있는 경우
                doc_content = text_list[:spot_list[idx]]
            elif idx == 0 and len(spot_list) == 1: 
                doc_prev_content = text_list[:spot_list[idx]]
                self.doc_no = self.doc_no + 1 
                doc_prev_text = ''.join(doc_prev_content)
                doc_prev = Document(text=doc_prev_text,
                           doc_id=f'{self.file_name}_doc_{self.doc_no}',
                            metadata={'spot': self.spot_point, 'file_name': self.file_name},
                            excluded_llm_metadata_key=['spot', 'file_name'])
                self.spot_point = split_spots[spot_list[idx]]
                self.new_doc.append(doc_prev)
                # splitted_doc.append(doc_prev)
                doc_content = text_list[spot_list[idx]+1:]
            elif idx + 1 == len(spot_list):
                doc_content = text_list[spot_list[idx]+1:]
                self.spot_point = split_spots[spot_list[idx]]
            else:
                doc_content = text_list[spot_list[idx]+1:spot_list[idx+1]]
                self.spot_point = split_spots[spot_list[idx+1]]       

            doc_text = ''.join(doc_content)
            self.doc_no = self.doc_no + 1
            doc = Document(text=doc_text,
                           doc_id=f'{self.file_name}_doc_{self.doc_no}',
                            metadata={'spot': self.spot_point, 'file_name': self.file_name},
                            excluded_llm_metadata_key=['spot', 'file_name'])
            print(f'prev_spot: {self.spot_point}')
            self.new_doc.append(doc)
    
    def create_new_doc(self, document):
        '''
        장 단위로 분할된 문서를 생성하는 함수 
        '''
        split_spot = self.get_split_spots(document)
        if len(split_spot) == 0:
            self.doc_no = self.doc_no + 1 
            doc = Document(text=document.text,
                           doc_id=f'{self.file_name}_doc_{self.doc_no}',
                            metadata={'spot': self.spot_point, 'file_name': self.file_name},
                            excluded_llm_metadata_key=['spot', 'file_name'])
            self.new_doc.append(doc)
        else:   # 
            self.split_doc(split_spot, document)
        
    def create_new_documents(self, old_documents):
        '''
        새로운 문서 생성하여 반환하는 함수 
        '''
        new_docs = []
        for idx, old_doc in enumerate(old_documents):
            if idx >= self.s_point:
                self.create_new_doc(old_doc)
       
    def process_table(self, document):
        '''
        table 요소를 전처리하는 함수 
        '''

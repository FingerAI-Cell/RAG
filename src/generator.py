import torch 
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig


class ResponseGenerator():
    def __init__(self, config):
        if config['model_type'] == 'mistral':
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(config['model_path'], 'mistral_origin', 'tokenizer'))
            self.model = AutoModelForCausalLM.from_pretrained(os.path.join(config['model_path'], 'mistral_origin'), \
                                                                  torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:0")
        elif config['model_type'] == 'llama2':
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(config['model_path'], 'llama2_origin', 'tokenizer'))
            self.model = AutoModelForCausalLM.from_pretrained(os.path.join(config['model_path'], 'llama2_origin'), torch_dtype=torch.float16, \
                                                              low_cpu_mem_usage=True, device_map="cuda:0")
        else:
            print('llama or mistral 모델만 지원합니다.')

    def set_gpu(self):
        self.g_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"    
        self.model.to(self.g_device)

    def set_generation_config(self):
        self.gen_config = GenerationConfig(
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            max_new_tokens=512,
        ) 

    def set_prompt_template(self, query, context):
        '''self.prompt_template = (
            f"""<s>[INST] 사용자가 물어본 내용은 '{query}'이거고, \n
            우리가 알고 있는 정보는 '{context}'이거야. \n 
            우리가 알고 있는 정보를 바탕으로 질문에 친절하게 답해줘 [/INST] \n"""
        )'''
        self.prompt_template = (
            f"""
            ### <s> [INST]
            참고: 다음 질문에 대해 너의 금융 정보에 기반해서 답을 해줘. 참고할만한 정보는 다음과 같아. 

            {context}

            ### Question:
            {query}
            [/INST] """
        )
    
    def generate_response(self):
        gened = self.model.generate(
            **self.tokenizer(
                self.prompt_template,
                return_tensors='pt',
                return_token_type_ids=False
            ).to(self.g_device),
            generation_config=self.gen_config,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # streamer=streamer,
        )
        
        result_str = self.tokenizer.decode(gened[0])
        start_tag = f"[/INST]"
        start_index = result_str.find(start_tag)

        if start_index != -1:
            response = result_str[start_index + len(start_tag):].strip()
        return response

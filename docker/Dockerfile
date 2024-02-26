# 기본 Docker Image 설정 
FROM pytorch/pytorch:latest
LABEL maintainer='ojd9512@gmail.com'
LABEL version='1.0.0'
LABEL description='Docker image created for rag architecture'

# Docker 내 작업 경로 설정 
WORKDIR /rag  
COPY requirements.txt .

# 한글 입력을 위한 환경 변수 설정 
ENV LC_ALL ko_KR.UTF-8 

# 포트 오픈 
EXPOSE 8888 

# Anaconda Install  -  pass 

# 패키지 설치 (한글 입력을 위한 locale)
RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8   
RUN apt-get install python3-pip -y
RUN pip install jupyter notebook
RUN jupyter notebook --generate-config 
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_conifg.py
RUN pip install -r requirements.txt
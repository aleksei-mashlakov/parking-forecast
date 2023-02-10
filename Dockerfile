FROM frolvlad/alpine-miniconda3
#FROM python:3.7

#RUN pip install --upgrade pip
RUN conda update conda

#COPY requirements.txt /requirements.txt
COPY ./src /app
COPY ./save /save
COPY ./logs /logs
COPY conda_env.yml /conda_env.yml

RUN conda env create -f /conda_env.yml && conda clean -afy
RUN echo "source activate PMV4Cast" > ~/.bashrc
ENV PATH /opt/conda/envs/PMV4Cast/bin:$PATH

#RUN pip install -r /requirements.txt

WORKDIR /app

ENTRYPOINT ["python", "/app/PMV4Cast/pipeline.py"]

FROM continuumio/miniconda3
USER root
#WORKDIR ./condatest
# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
RUN echo "conda activate sahi" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
# Demonstrate the environment is activated:
RUN echo "Make sure OpenCV is installed:"
RUN python -c "import cv2"
RUN pip install cython_bbox
# The code to run when container is started:
WORKDIR ./real_deal
RUN pwd
#COPY ./f2/run.py ./f2/entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]

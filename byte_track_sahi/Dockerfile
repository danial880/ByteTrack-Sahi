FROM continuumio/miniconda3
USER root
WORKDIR home
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
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
RUN git clone https://github.com/ifzhang/ByteTrack.git
WORKDIR ByteTrack
RUN python setup.py install
#COPY ./f2/run.py ./f2/entrypoint.sh ./
RUN  chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

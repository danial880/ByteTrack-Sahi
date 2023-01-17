FROM continuumio/miniconda3
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
RUN echo "conda activate sahi" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
RUN pip install cython_bbox
RUN git clone https://github.com/ifzhang/ByteTrack.git
#RUN cd ByteTrack && python setup.py develop
RUN pip install -r https://raw.githubusercontent.com/WongKinYiu/yolov7/main/requirements.txt
RUN git clone https://github.com/danial880/Sahi-Yolov7.git
RUN cd Sahi-Yolov7 && python setup.py install
#RUN git clone https://github.com/danial880/ByteTrack-Sahi.git
RUN ls
RUN pip install lap
RUN git clone https://github.com/danial880/ByteTrack-Sahi.git
RUN cd ByteTrack-Sahi/byte_track_sahi && python setup.py install
WORKDIR /byte_track_sahi
ENTRYPOINT ["./entrypoint.sh"]


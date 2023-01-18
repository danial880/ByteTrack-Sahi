# ByteTrack-Sahi
## Download Models
- Download [Yolov7-E6E](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) and put it under [yolo_models](https://github.com/danial880/ByteTrack-Sahi/tree/main/byte_track_sahi/yolo_models) folder
- Download [bytetrack_x_mot20](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing) and put it under [pretrained](https://github.com/danial880/ByteTrack-Sahi/tree/main/byte_track_sahi/pretrained) folder

<details>

<summary>
<big><b>Nvidia-Docker2 Installation</b></big>
</summary>  
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.listdir
```

```
sudo apt-get update
```
```
sudo apt-get install -y nvidia-docker2
```
```
sudo systemctl restart docker
```
</details>  

__Note:__ Docker-compose version >=1.28  


<details>

<summary>
<big><b>Docker</b></big>
</summary>  

- Install
```js
  sudo docker-compose up --build
```
- Run
```js
  sudo docker-compose up 
```
- Close
```js
  sudo docker-compose down
```
</details> 

## Inference
#### Set Path
Open byte_track_sahi/tools/config.py
```
# at line number 5
path = 'path/to/image/directory'
```
### ByteTrack
Open byte_track_sahi/tools/config.py
```
# at line number 3
model_to_run = 'norm_yoloX'
```
### ByteTrack with SAHI (Yolov7)
Open byte_track_sahi/tools/config.py
```
# at line number 3
model_to_run = 'sahi_yolo7'
```
### ByteTrack with SAHI (YolovX)
Open byte_track_sahi/tools/config.py
```
# at line number 3
model_to_run = 'sahi_yoloX'
```

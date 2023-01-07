# ByteTrack-Sahi
## Download Models
- Download [Yolov7-E6E](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) and put it under [yolo_models](https://github.com/danial880/ByteTrack-Sahi/tree/main/byte_track_sahi/yolo_models) folder
- Download [bytetrack_x_mot20](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing) and put it under [pretrained](https://github.com/danial880/ByteTrack-Sahi/tree/main/byte_track_sahi/pretrained) folder
<details>

<summary>
<big><b>Docker Installation</b></big>
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

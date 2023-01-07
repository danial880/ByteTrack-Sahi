class Config(object):
    # model to run options('sahi_yolo7','sahi_yolox','norm_yolox')
    model_to_run = 'sahi_yolo7'
    # path to image directory
    path = 'train_part7/07_University_Campus'#'path/to/image/directory'
    # model name
    name = 'yolox'
    # name of experiment
    experiment_name = 'sahi_track_yolov7'
    # demo type, e.g. image, video and webcam
    demo = 'image'
    # whether to save the inference results in images
    save_result = False
    # whether to save tracking results in a txt
    save_txt = True
    # expriment description file
    exp_file = 'exps/example/mot/yolox_x_mix_mot20_ch.py'
    # check point for evaluation
    ckpt = 'pretrained/bytetrack_x_mot20.tar'
    # path to yolov7 model
    yolov7 = 'yolo_models/yolov7-e6e.pt'
    # device to run our model e.g. cpu or gpu
    device = 'gpu'
    # test mot20
    mot20 = False
    # 
    fp16 = False
    ########################## detection parameters ############################
    # test conf
    conf = 0.25
    # test img size
    tsize = 5120
    # nms threshold
    nms = 0.45
    ########################## tracking parameters #############################
    # tracking confidence threshold
    track_thresh = 0.50
    # frame rate
    fps = 30
    # filter out tiny boxes
    min_box_area = 10
    # threshold for filtering out boxes
    aspect_ratio_thresh = 1.6
    # matching threshold for tracking
    match_thresh = 0.8
    # the frames for keep lost tracks
    track_buffer = 30
    # SAHI metric(intersection over samll area)
    ios = 0.8
    #sahi overlap height ratio
    overlap_height = 0.1
    #sahi overlap width ratio
    overlap_width = 0.1

    

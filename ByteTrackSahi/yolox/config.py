class ConfigVX(object):
    # path to image directory
    path = 'path/to/image/directory'
    # name of experiment
    experiment_name = 'sahi_track_yolovX'
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
    ########################## tracking parameters ############################
    # tracking confidence threshold
    track_thresh = 0.50
    # frame rate
    fps = 30
    # threshold for filtering out boxes of which aspect ratio are above the given value.
    aspect_ratio_thresh = 1.6
    # matching threshold for tracking
    match_thresh = 0.8
    #
    track_buffer = 30
    

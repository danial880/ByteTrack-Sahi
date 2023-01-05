class ConfigVX(object):
    # name of experiment
    exp_name = 'sahi_track_yolovX'
    # demo type, e.g. image, video and webcam
    demo_type = 'image'
    # whether to save the inference results in images
    save_result = False
    # whether to save tracking results in a txt
    save_txt = True
    # expriment description file
    exp_file = 'exps/example/mot/yolox_x_mix_mot20_ch.py'
    # check point for evaluation
    check_point = 'pretrained/bytetrack_x_mot20.tar'
    # device to run our model e.g. cpu or gpu
    device = 'gpu'
    ########################## detection parameters ############################
    # test conf
    conf = 0.25
    # test img size
    tsize = 5120
    #
    

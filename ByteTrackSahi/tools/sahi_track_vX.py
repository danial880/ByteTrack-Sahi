import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
from sahi.prediction import ObjectPrediction
from sahi.postprocess.utils import ObjectPredictionList, has_match
from sahi.postprocess.utils import merge_object_prediction_pair
from sahi.slicing import slice_image
from sahi.postprocess.combine import batched_greedy_nmm
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--demo", default="image",
        help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None,
        help="model name")
    parser.add_argument("--path", default="./videos/palace.mp4",
        help="path to images or video")
    parser.add_argument("--camid", type=int, default=0,
        help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",
        help="whether to save the inference result of image/video")    
    parser.add_argument("--save_txt", action="store_true",
        help="whether to save the tracking result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str,
        help="pls input your expriment description file",)
    parser.add_argument("-c", "--ckpt", default=None, type=str,
        help="ckpt for eval")
    parser.add_argument("--device", default="gpu", type=str,
        help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float,
        help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False,
        action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False,
        action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true",
        help="Using TensorRT model for testing.")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.47,
        help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30,
        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8,
        help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10,
        help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False,
        action="store_true", help="test mot20.")
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def get_slices(self, img):
        slice_image_result = slice_image(
            image=img,
            slice_height=exp.test_size[0],
            slice_width=exp.test_size[0],
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            auto_slice_resolution=False,
        )
        return slice_image_result

    def get_lists(self, selected_object_predictions):
        bbox_list = []
        score_list = []
        for pred in selected_object_predictions:
            if pred.category.id == 0:
                bbox = pred.bbox.to_voc_bbox()
                score = pred.score.value
                bbox_list.append(bbox)
                score_list.append(score)
        return bbox_list, score_list

    def sahi_postproc(self, tensor, object_prediction_list):
        cuda_tensor = tensor.cuda()
        keep_to_merge_list = batched_greedy_nmm(
            cuda_tensor,
            match_metric = "IOS",
            match_threshold = 0.8,
        )
        object_prediction_list = ObjectPredictionList(object_prediction_list)
        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    "IOS",
                    0.8,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(),
                        object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())
        
        return self.get_lists(selected_object_predictions)

    def bytetrack_preproc(self, img):
        proc_img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        proc_tensor = torch.from_numpy(proc_img).unsqueeze(0).float().to(self.device)
        return proc_tensor

    def get_model_output(self, proc_tensor):
        with torch.no_grad():
            outputs = self.model(proc_tensor)
            outputs = postprocess(outputs, self.num_classes,
                                self.confthre,self.nmsthre)
        return outputs

    def get_sorted_output(self, output):
        output = output.cpu().numpy()
        output[:, 4] = output[:, 4] * output[:, 5]
        output = np.delete(output,5,axis=1)
        return output

    def get_scaled_output(self, output, scale):
        for i in range(4):
            output[:, i] = output[:, i] / scale
        return output 

    def get_shifted_output(self, output, shift):
        index = [0, 1, 0, 1]
        for i in range(4):
            output[:, i] = output[:, i] + shift[index[i]]
        return output 

    def get_pred_list(self, output):
        pred_list = []
        for pred in output:
            x1, y1, x2, y2 = (int(pred[0]), int(pred[1]), int(pred[2]),
                            int(pred[3]),)
            bbox = [x1, y1, x2, y2]
            score = pred[4]
            category_id = pred[5]
            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=int(category_id),
                score=score,
                bool_mask=None,
                category_name='person',
                shift_amount=[0, 0],
                full_shape=None,
            )
            pred_list.append(object_prediction)
        return pred_list

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["raw_img"] = img
        #inference on full image
        object_prediction_list = []
        proc_tensor_full = self.bytetrack_preproc(img)
        outputs_full = self.get_model_output(proc_tensor_full)
        if outputs_full[0] is not None:
            output_results = self.get_sorted_output(outputs_full[0])
            scale = min(self.test_size[0] / float(height),
                        self.test_size[0] / float(width))
            output_scaled = self.get_scaled_output(output_results, scale)
            pred_list_full = self.get_pred_list(output_scaled)
            object_prediction_list.extend(pred_list_full)
            full_tensor = torch.from_numpy(output_scaled)           
        slice_image_result = self.get_slices(img)
        slices = slice_image_result.images
        previous = False
        for count, slicee in enumerate(slices):
            shift = slice_image_result.starting_pixels[count]
            tensor_proc_slice = self.bytetrack_preproc(slicee)
            outputs = self.get_model_output(tensor_proc_slice)
            if outputs[0] is not None:
                output_results = self.get_sorted_output(outputs[0])
                shifted_output = self.get_shifted_output(output_results, shift)                            
                new_tensor = torch.from_numpy(shifted_output)
                pred_list_slice = self.get_pred_list(shifted_output)
                object_prediction_list.extend(pred_list_slice)
                if not previous:
                    current_tensor = torch.cat((full_tensor, new_tensor))
                    previous = True
                else:
                    big_tensor = torch.cat((current_tensor, new_tensor))
                    current_tensor = big_tensor                
        print('Before SAHI post-processing',current_tensor.size())
        bbox_list, score_list = self.sahi_postproc(current_tensor,
                                                object_prediction_list)
        print('after sahi post_processing',len(bbox_list))
        
        return bbox_list, score_list, img_info

def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(tqdm(files), 1):
        bbox, score, img_info = predictor.inference(img_path, timer)
        bbox_numpy = np.array(bbox)
        score_numpy = np.array(score)
        if bbox_numpy is not None:
            online_targets = tracker.update_sahi(bbox_numpy, score_numpy)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{int(tlwh[0])},{int(tlwh[1])},{int(tlwh[2])},{int(tlwh[3])},-1,-1,-1,-1\n"
                    )
            timer.toc()
            if args.save_result:
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
                )
        else:
            timer.toc()
            online_im = img_info['raw_img']
        # save images
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)
        # log info every 20th frame
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # exit 
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    # save trackings in a txt
    if args.save_txt:
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result or args.save_txt :
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    model.eval()

    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")


    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)



if __name__ == "__main__":
    args = make_parser().parse_args()
    #print(args.name)
    exp = get_exp(args.exp_file, args.name)
    #print(exp)
    main(exp, args)

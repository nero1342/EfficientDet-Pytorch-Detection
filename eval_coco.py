# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.backbone import EfficientDetBackbone
from models.efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from utils.random_seed import set_seed, set_determinism
from utils.getter import get_instance, get_data
from utils.device import move_to

import argparse
import pprint

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs([image_id])[0]
        image_path = os.path.join(img_path, image_info['file_name'])
        
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path)
        x = torch.from_numpy(framed_imgs[0])

        x = move_to(x, args.gpus)
        # x = x.cuda(gpu)
        x = x.float()
        
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['debug'] = args.debug
    config['gpus'] = args.gpus
    
    set_determinism()
    assert config is not None, "Do not have config file!"

    pprint.PrettyPrinter(indent=2).pprint(config)

    # Get device
    dev_id = 'cuda:{}'.format(config['gpus']) \
        if torch.cuda.is_available() and config.get('gpus', None) is not None \
        else 'cpu'
    device = torch.device(dev_id)

    # Get pretrained model
    pretrained_path = config["pretrained"]

    pretrained = None
    if (str(pretrained_path) != 'None'):
        pretrained = torch.load(pretrained_path, map_location=dev_id)
        # for item in ["model"]:
        #     config[item] = pretrained["config"][item]

    # 1: Load datasets
    _, val_dataloader = \
        get_data(config['dataset'], config['seed'])

    # 2: Define network
    model = get_instance(config['model']).to(device)

    # Train from pretrained if it is not None
    if pretrained is not None:
        try:
            ret = model.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

    SET_NAME = config['dataset']['val']['args']['set']
    VAL_IMGS = config['dataset']['val']['args']['img_dir']
    VAL_GT = config['dataset']['val']['args']['path_to_json']
    MAX_IMAGES = 100
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    threshold = config['threshold']
    obj_list = config['obj_list']

    
    model.requires_grad_(False)
    model.eval()
    
    evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')

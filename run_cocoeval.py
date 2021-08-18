import numpy as np
import torch
import json
import torchvision.transforms as transforms
from utils_box.dataset import Dataset_CSV
from utils_box.eval_csv import eval_detection
from detector import Detector, get_loss, get_pred
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os


with open('./data/voc_name.txt', 'r') as f:
    name_list = f.readlines()
    name_list2 = [line.strip('\n') for line in name_list]

# TODO: Set your coco_table_file, coco_anno_root and set_name
coco_table_file = 'data/coco_table.json'
coco_anno_root = 'data/'
set_name = 'val2017'
# ==================================


# Read train.json/coco_table_file and set current GPU (for nms_cuda)
with open(coco_table_file, 'r') as load_f:
    coco_table = json.load(load_f)
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])

# Prepare the network
net = Detector(pretrained=False)
device_out = 'cuda:%d' % (cfg['device'][0])
net.load_state_dict(torch.load('tsign_sc_45.pkl', map_location=device_out))
net = net.cuda(cfg['device'][0])
net.eval()


# Get eval dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.546, 0.594, 0.642), (0.187, 0.195, 0.232))])
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.eval_size, train=False, transform=transform)


# Eval
with torch.no_grad():
    results = []
    for i in range(len(dataset_eval)):

        img, bbox, label, scale = dataset_eval[i]
        img = img.cuda(cfg['device'][0])
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])

        temp = net(img)
        cls_i_preds, cls_p_preds, reg_preds = get_pred(temp, 
                net.nms_th, net.nms_iou, net.eval_size)
                
        cls_i_preds = cls_i_preds[0].cpu()
        cls_p_preds = cls_p_preds[0].cpu()
        reg_preds = reg_preds[0].cpu()

        if reg_preds.shape[0] > 0:

            ymin, xmin, ymax, xmax = reg_preds.split([1, 1, 1, 1], dim=1)
            h = ymax - ymin
            w = xmax - xmin 
            reg_preds = torch.cat([xmin, ymin, w, h], dim=1)
            reg_preds = reg_preds / float(scale)

            for box_id in range(reg_preds.shape[0]):

                score = float(cls_p_preds[box_id])
                label = int(cls_i_preds[box_id])
                box = reg_preds[box_id, :]

                image_result = {
                    'image_id'    : coco_table['val_image_ids'][i],
                    'category_id' : coco_table['coco_labels'][str(label)],
                    'score'       : float(score),
                    'bbox'        : box.tolist(),
                }

                results.append(image_result)

        print('step:%d/%d' % (i, len(dataset_eval)), end='\r')

    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)

    coco = COCO(os.path.join(coco_anno_root, 'instances_' + set_name + '.json'))
    coco_pred = coco.loadRes('coco_bbox_results.json')

    # run COCO evaluation
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.params.imgIds = coco_table['val_image_ids']
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
#    pr_array1 = coco_eval.eval['precision'][0, :, :, 0, 2]
#    pr_array2 = coco_eval.eval['precision'][0, :, :, 0, 2]
#    pr_array3 = coco_eval.eval['precision'][0, :, :, 0, 2]
    #
#    for i in range(0, 45):
#       small = open('pr/all/small/pn_s_' + name_list2[i + 1] + '.txt', 'a')
#       medium = open('pr/all/med/pn_m_' + name_list2[i + 1] + '.txt', 'a')
#       large = open('pr/all/large/pn_l_' + name_list2[i + 1] + '.txt', 'a')
#       ss = list(pr_array1[:, i])
#       mm = list(pr_array2[:, i])
#       ll = list(pr_array3[:, i])
#       for s0 in ss:
#           small.write(str(s0) + '\n')
#       for m0 in mm:
 #          medium.write(str(m0) + '\n')
 #      for l0 in ll:
#           large.write(str(l0) + '\n')
    # pr_array4 = coco_eval.eval['recall'][0, :, 1, 2]
    # pr_array5 = coco_eval.eval['recall'][0, :, 2, 2]
    # pr_array6 = coco_eval.eval['recall'][0, :, 3, 2]
    #
    # for j in range(0, 45):
    #     small_r = open('recall/small/r_s50_' + name_list2[j + 1] + '.txt', 'a')
    #     medium_r = open('recall/med/r_m50_' + name_list2[j + 1] + '.txt', 'a')
    #     large_r = open('recall/large/r_l50_' + name_list2[j + 1] + '.txt', 'a')
    #     print(pr_array4.shape)
    #     ss1 = pr_array4[j]
    #     mm1 = pr_array5[j]
    #     ll1 = pr_array6[j]
    #
    #     small_r.write(str(ss1) + '\n')
    #
    #     medium_r.write(str(mm1) + '\n')
    #
    #     large_r.write(str(ll1) + '\n')



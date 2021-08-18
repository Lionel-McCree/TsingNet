import numpy as np
import torch
import json
import torchvision.transforms as transforms
from utils_box.dataset import Dataset_CSV
from utils_box.eval_csv import eval_detection
from detector import Detector, get_loss, get_pred

import pickle as pkl
# 修改部分2
import matplotlib.pyplot as plt
with open('./data/voc_name.txt', 'r') as f:
    name_list = f.readlines()
    name_list2 = [line.strip('\n') for line in name_list]

# Read train.json and set current GPU (for nms_cuda)
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])
net = Detector(pretrained=False)


# TODO: You can change nms_th, device or nbatch_eval
# net.nms_th = 0.05
# cfg['device'] = [0,1,2,3,9]
# cfg['nbatch_eval'] = 30
# ==================================


# Prepare the network
device_out = 'cuda:%d' % (cfg['device'][0])
net.load_state_dict(torch.load('tsign_sc_45.pkl', map_location=device_out))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.eval()


# Get eval dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.546,0.594, 0.642), (0.187, 0.195,0.232))])
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.module.eval_size, train=False, transform=transform)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=cfg['nbatch_eval'], 
                    shuffle=False, num_workers=0, collate_fn=dataset_eval.collate_fn)


# Eval
with torch.no_grad():
    net.eval()
    pred_bboxes = []
    pred_labels = []
    pred_scores = []
    gt_bboxes = []
    gt_labels = []
    for i, (img, bbox, label, scale) in enumerate(loader_eval):
        temp = net(img)
        cls_i_preds, cls_p_preds, reg_preds = get_pred(temp, 
                net.module.nms_th, net.module.nms_iou, net.module.eval_size)
        for idx in range(len(cls_i_preds)):
            cls_i_preds[idx] = cls_i_preds[idx].cpu().detach().numpy()
            cls_p_preds[idx] = cls_p_preds[idx].cpu().detach().numpy()
            reg_preds[idx] = reg_preds[idx].cpu().detach().numpy()
        _boxes = []
        _label = []
        for idx in range(bbox.shape[0]):
            mask = label[idx] > 0
            _boxes.append(bbox[idx][mask].detach().numpy())
            _label.append(label[idx][mask].detach().numpy())
        pred_bboxes += reg_preds
        pred_labels += cls_i_preds
        pred_scores += cls_p_preds
        gt_bboxes += _boxes
        gt_labels += _label
        print('  Eval: {}/{}'.format(i*cfg['nbatch_eval'], len(dataset_eval)), end='\r')
    # ap_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ap_iou = [0.5]
    ap_res = []
    for iou_th in ap_iou:
        res = eval_detection(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_th=iou_th)
        ap_res.append(res)

    map_50 = float(ap_res[0]['map'])
    print('map_50')
    print(map_50)
        # recall2 = []
        # prec2 = []
        # if iou_th == 0.5:
        #     prec = prec[1:46]
        #     rec = rec[1:46]
        #     label_name = name_list2[1:46]
        #     # for j in range(len(prec)):
        #     #     pkl_path = open('./large/' + label_name[j] + '.pkl', 'wb')
        #     #     pkl.dump({'rec': rec[j], 'prec': prec[j]}, pkl_path)
        # map_50 = float(ap_res[0]['map'])
        # print(map_50)
    ap = ap_res[0]['ap']
    ap_txt = open('./ap/ap.txt', 'a')
    ap = ap[1:]
    label_name = name_list2[1:46]
    for i in range(len(ap)):
        ap_txt.write(str(ap[i]) + '\n')
        print(label_name[i] + ':' + str(ap[i]))

        # print(np.mean(prec2))
        # print(np.mean(recall2))
        # print('mAP:', float(res['map']))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.plot(rec[1:46], prec[1:46])
        # plt.show()
                # name = label_name[j]
                # print(name + '-prec', np.mean(prec[j]))
                # print(name + '-recall', np.mean(rec[j]))

        # ----------原始代码部分
        # ap_res.append(res)

    # ap_sum = 0.0
    # for i in range(len(ap_res)):
    #     ap_sum += float(ap_res[i]['map'])
    # map_mean = ap_sum / float(len(ap_res))
    # map_50 = float(ap_res[0]['map'])
    # map_75 = float(ap_res[5]['map'])
    #
    # print('map_mean')
    # print(map_mean)
    # print('map_50')
    # print(map_50)
    # print('map_75')
    # print(map_75)
    # print('ap@.5')
    # print(ap_res[0]['ap'])

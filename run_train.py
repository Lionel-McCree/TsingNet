import numpy as np
import torch
import json
import time
import torchvision.transforms as transforms
from utils_box.dataset import Dataset_CSV
from utils_box.eval_csv import eval_detection
from detector import Detector, get_loss, get_pred

# Read train.json and set current GPU (for nms_cuda)/home/ff/data/DFG/demo
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])

# Prepare the network and read log
net = Detector(pretrained=cfg['pretrain'])
log = []
device_out = 'cuda:%d' % (cfg['device'][0])
if cfg['load']:
    net.load_state_dict(torch.load('tsign_sc_28.pkl', map_location=device_out))
    log = list(np.load('tsign_sc.npy'))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
torch.backends.cudnn.benchmark = True
net.train()

# Get train/eval dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.546, 0.594, 0.642), (0.187, 0.195, 0.232))])
dataset_train = Dataset_CSV(cfg['root_train'], cfg['list_train'], cfg['name_file'],
                            size=net.module.train_size, train=True, transform=transform,
                            boxarea_th=cfg['boxarea_th'],
                            img_scale_min=cfg['img_scale_min'],
                            crop_scale_min=cfg['crop_scale_min'],
                            aspect_ratio=cfg['aspect_ratio'],
                            remain_min=cfg['remain_min'],
                            augmentation=cfg['augmentation'])
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'],
                           size=net.module.eval_size, train=False, transform=transform)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg['nbatch_train'],
                                           shuffle=True, num_workers=cfg['num_workers'],
                                           collate_fn=dataset_train.collate_fn)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=cfg['nbatch_eval'],
                                          shuffle=False, num_workers=0, collate_fn=dataset_eval.collate_fn)

# Prepare optimizer
lr = cfg['lr']
lr_decay = cfg['lr_decay']
opt = torch.optim.SGD(net.parameters(), lr=lr,
                      momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

# Run warmup
#WARM_UP_ITERS = 500
#WARM_UP_FACTOR = 1.0 / 3.0
#if cfg['freeze_bn']:
#   net.module.backbone.freeze_bn()
#for i, (img, bbox, label, scale) in enumerate(loader_train):
#   alpha = float(i) / WARM_UP_ITERS
#   warmup_factor = WARM_UP_FACTOR * (1.0 - alpha) + alpha
#   for param_group in opt.param_groups:
#       param_group['lr'] = lr * warmup_factor
#   time_start = time.time()
#   opt.zero_grad()
#   temp, scale_loss = net(img, label, bbox)
#   loss1 = get_loss(temp)
#   loss = loss1 + scale_loss
#   loss.backward()
#   clip = cfg['grad_clip']
#   torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
#   opt.step()
#   maxmem = int(torch.cuda.max_memory_allocated(device=cfg['device'][0]) / 1024 / 1024)
#   time_end = time.time()
#   totaltime = int((time_end - time_start) * 1000)
#   print('warmup: step:%d/%d, lr:%f, loss_c_r:%f, loss_s:%f, maxMem:%dMB, time:%dms' % \
#         (i, WARM_UP_ITERS, lr * warmup_factor, loss1, scale_loss, maxmem, totaltime))
#   with open('tsign_sc.txt', 'a') as f:
#       f.write(str(loss.item()) + '\n')
#   if i >= WARM_UP_ITERS:
#       break


# Run epoch
epoch = 31
j = 31
for epoch_num in cfg['epoch_num']:  # 3 for example

    for param_group in opt.param_groups:
        param_group['lr'] = lr

    for e in range(epoch_num):
        if cfg['freeze_bn']:
            net.module.backbone.freeze_bn()

        # Train
        torch.set_num_threads(2)
        start = time.time()
        for i, (img, bbox, label, scale) in enumerate(loader_train):
            end = time.time()
            cpu_time = int((end - start) * 1000)
            start = end
            time_start = time.time()
            opt.zero_grad()
            temp, scale_loss = net(img, label, bbox)
            loss1 = get_loss(temp)
            loss = loss1 + scale_loss
            loss.backward()
            clip = cfg['grad_clip']
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=cfg['device'][0]) / 1024 / 1024)
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)

            if i * cfg['nbatch_train'] % 10 == 0:
                print('epoch:%d, step:%d/%d, loss_c:%f, loss_s:%f, cpu:%dms, time:%dms' % \
                      (epoch, i * cfg['nbatch_train'], len(dataset_train), loss1, scale_loss, cpu_time, totaltime))

            with open('tsign_sc1.txt', 'a') as f:
                f.write(str(loss1.item()) + ' ' +str(scale_loss.item()) + '\n')

            if (i + 1) % 10000 == 0:
                j += 1
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
                                                                      net.module.nms_th, net.module.nms_iou,
                                                                      net.module.eval_size)
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
                       print('  Eval: {}/{}'.format(i * cfg['nbatch_eval'], len(dataset_eval)), end='\r')
                   ap_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                   #ap_iou = [0.5, 0.75]
                   ap_res = []
                   for iou_th in ap_iou:
                       res = eval_detection(pred_bboxes, pred_labels,
                                            pred_scores, gt_bboxes, gt_labels, iou_th=iou_th)
                       ap_res.append(res)
                   ap_sum = 0.0
                   for i in range(len(ap_res)):
                       ap_sum += float(ap_res[i]['map'])
                   map_mean = ap_sum / float(len(ap_res))
                   map_50 = float(ap_res[0]['map'])
                   map_75 = float(ap_res[1]['map'])
                   print('map_mean:', map_mean, 'map_50:', map_50, 'map_75:', map_75)
                   log.append([map_mean, map_50, map_75])
                   net.train()

        # Save
                if cfg['save']:
                    torch.save(net.module.state_dict(), 'tsign_sc_' + str(j) + '.pkl')
                    if len(log) > 0:
                        np.save('tsign_sc.npy', log)

        epoch += 1

    lr *= lr_decay

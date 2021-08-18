import torch
import numpy as np 
import matplotlib.pyplot as plt



def box_iou(box1, box2, eps=1e-10):
    '''
    Param:
    box1:   FloatTensor(n,4) # 4: ymin, xmin, ymax, xmax
    box2:   FloatTensor(m,4)

    Return:
    FloatTensor(n,m)
    '''
    tl = torch.max(box1[:, None, :2], box2[:, :2])  # [n,m,2]
    br = torch.min(box1[:, None, 2:], box2[:, 2:])  # [n,m,2]
    hw = (br-tl+eps).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    area1 = (box1[:,2]-box1[:,0]+eps) * (box1[:,3]-box1[:,1]+eps)  # [n,]
    area2 = (box2[:,2]-box2[:,0]+eps) * (box2[:,3]-box2[:,1]+eps)  # [m,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


def gen_anchors(a_hw, scales, img_size, first_stride):

    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    anchors_yxyx = []
    anchors_yxhw = []
    stride = first_stride
    an = len(a_hw)
    for scale_id in range(scales):
        fsz_h = (img_size[0]) // (first_stride * pow(2, scale_id))
        fsz_w = (img_size[1]) // (first_stride * pow(2, scale_id))
        anchors_yxyx_i = torch.zeros(fsz_h, fsz_w, an, 4)
        anchors_yxhw_i = torch.zeros(fsz_h, fsz_w, an, 4)
        for h in range(fsz_h):
            for w in range(fsz_w):
                a_y, a_x = (h+0.5) * float(stride), (w+0.5) * float(stride)
                scale = float(stride//first_stride)
                for a_i in range(an):
                    a_h, a_w = scale*a_hw[a_i][0], scale*a_hw[a_i][1]
                    a_h_2, a_w_2 = a_h/2.0, a_w/2.0
                    a_ymin, a_ymax = a_y - a_h_2, a_y + a_h_2
                    a_xmin, a_xmax = a_x - a_w_2, a_x + a_w_2
                    anchors_yxyx_i[h, w, a_i, :] = \
                        torch.Tensor([a_ymin, a_xmin, a_ymax, a_xmax])
                    anchors_yxhw_i[h, w, a_i, :] = \
                        torch.Tensor([a_y, a_x, a_h, a_w])
        stride *= 2
        anchors_yxyx_i = anchors_yxyx_i.view(fsz_h*fsz_w*an, 4)
        anchors_yxhw_i = anchors_yxhw_i.view(fsz_h*fsz_w*an, 4)
        anchors_yxyx.append(anchors_yxyx_i)
        anchors_yxhw.append(anchors_yxhw_i)
    return torch.cat(anchors_yxyx, dim=0), torch.cat(anchors_yxhw, dim=0)



if __name__ == '__main__':
    A_HW = [
        [14.0, 14.0],
        [12.8, 19.6],
        # [39.6, 19.8]
    ]
    SCALES        = 3
    IMG_SIZE    = (80,300)
    FIRST_STRIDE  = 64
    ANCHORS_YXYX, ANCHORS_YXHW = gen_anchors(A_HW, SCALES, IMG_SIZE, FIRST_STRIDE)
    print(ANCHORS_YXYX.shape)
    print(ANCHORS_YXHW.shape)
    img_1 = torch.zeros(IMG_SIZE[0], IMG_SIZE[1])
    img_2 = torch.zeros(IMG_SIZE[0], IMG_SIZE[1])

    for n in range(ANCHORS_YXYX.shape[0]):
        ymin, xmin, ymax, xmax = ANCHORS_YXYX[n]
        y, x, h, w = ANCHORS_YXHW[n]

        ymin = torch.clamp(ymin, min=0, max=IMG_SIZE[0]-1)
        xmin = torch.clamp(xmin, min=0, max=IMG_SIZE[1]-1)
        ymax = torch.clamp(ymax, min=0, max=IMG_SIZE[0]-1)
        xmax = torch.clamp(xmax, min=0, max=IMG_SIZE[1]-1)
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax) 
        img_1[ymin, xmin:xmax] = 1.0
        img_1[ymax, xmin:xmax] = 1.0
        img_1[ymin:ymax, xmin] = 1.0
        img_1[ymin:ymax, xmax] = 1.0

        _ymin = y - h/2.0
        _xmin = x - w/2.0
        _ymax = y + h/2.0
        _xmax = x + w/2.0
        _ymin = torch.clamp(_ymin, min=0, max=IMG_SIZE[0]-1)
        _xmin = torch.clamp(_xmin, min=0, max=IMG_SIZE[1]-1)
        _ymax = torch.clamp(_ymax, min=0, max=IMG_SIZE[0]-1)
        _xmax = torch.clamp(_xmax, min=0, max=IMG_SIZE[1]-1)
        _ymin, _xmin, _ymax, _xmax = int(_ymin), int(_xmin), int(_ymax), int(_xmax) 
        img_2[_ymin, _xmin:_xmax] = 1.0
        img_2[_ymax, _xmin:_xmax] = 1.0
        img_2[_ymin:_ymax, _xmin] = 1.0
        img_2[_ymin:_ymax, _xmax] = 1.0

    plt.subplot(1,2,1)
    plt.imshow(img_1.numpy())
    plt.subplot(1,2,2)
    plt.imshow(img_2.numpy())
    plt.show()

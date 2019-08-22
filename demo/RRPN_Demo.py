import os
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg
from demo.predictor import ICDARDemo, RRPNDemo
from maskrcnn_benchmark.utils.visualize import vis_image, write_result_ICDAR_RRPN2polys, zip_dir
from PIL import Image
import time

from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms  # clw modify
########################################################
NAME_LABEL_MAP = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15,
        'helipad':16, #clw modify
        'airport':17, #clw modify
        'container-crane':18  #clw modify
    }

def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()
################################################################3


#config_file = "./configs/rrpn/e2e_rrpn_R_50_C4_1x_ICDAR13_15_17_trial_again_test.yaml"
#config_file = "../configs/rrpn/e2e_rrpn_R_50_C4_1x_DOTA_test.yaml"
config_file = "/home/clwclw/RRPN_pytorch_test/configs/rrpn/e2e_rrpn_X_101_32x8d_FPN_1x_DOTA.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.freeze()
# cfg.MODEL.WEIGHT = 'models/IC-13-15-17-Trial/model_0155000.pth'

result_dir = os.path.join('results', config_file.split('/')[-1].split('.')[0], cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0])

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

vis = False
if vis == True:
    confidence_threshold = 0.5
else:
    confidence_threshold = 0.01

#dataset_name = 'IC15'
dataset_name = 'DOTA'

testing_dataset = {
    # 'IC13': {
    #     'testing_image_dir': '../datasets/ICDAR13/Challenge2_Test_Task12_Images',
    #     'test_vocal_dir': '../datasets/ICDAR13/ch2_test_vocabularies_per_image'
    # },
    # 'IC15': {
    #     'testing_image_dir': '../datasets/ICDAR15/ch4_test_images',
    #     'test_vocal_dir': '../datasets/ICDAR15/ch4_test_vocabularies_per_image'
    # },
    'DOTA': {
        #'testing_image_dir': '/media/clwclw/data/DOTA_15classes/val_test/images',
        'testing_image_dir': '/home/clwclw/test',
        #'testing_image_dir': '/media/clwclw/Elements/deep_learning/competion/2019yaogan/test',
        'test_vocal_dir': '../datasets/ICDAR15/ch4_test_vocabularies_per_image'
    }
}

coco_demo = RRPNDemo(
    cfg,
    min_image_size=800,
    #confidence_threshold=0.85,
    confidence_threshold=confidence_threshold,
)

image_dir = testing_dataset[dataset_name]['testing_image_dir']
vocab_dir = testing_dataset[dataset_name]['test_vocal_dir']

# load image and then run prediction
# image_dir = '../datasets/ICDAR13/Challenge2_Test_Task12_Images/'
imlist = os.listdir(image_dir)

print('************* META INFO ***************')
print('config_file:', config_file)
print('result_dir:', result_dir)
print('image_dir:', image_dir)
print('weights:', cfg.MODEL.WEIGHT)
print('***************************************')



num_images = len(imlist)
cnt = 0


##### 如果不需要crop,用这个方法
# for image in imlist:
#     impath = os.path.join(image_dir, image)
#     # print('image:', impath)
#     img = cv2.imread(impath)
#     cnt += 1
#     tic = time.time()
#     predictions, bounding_boxes = coco_demo.run_on_opencv_image(img)
#     ##########################################
#     #clw modify: 测试使用
#     #import matplotlib.pyplot as plt
#     #plt.imshow(predictions[:, :, [2, 1, 0]])
#     #plt.axis("off")
#     #plt.show()
#     ##########################################
#     toc = time.time()
#
#     print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
#     #bboxes_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN
#
#     width, height = bounding_boxes.size
#
#     if vis:
#         #pil_image = vis_image(Image.fromarray(img), bboxes_np)
#         predictions = cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB)
#         pil_image = vis_image(Image.fromarray(predictions), bboxes_np) # clw modify
#         pil_image.save(image)
#         #pil_image.show()
#         #time.sleep(1)  # clw modify: 之前是10
#     write_result_ICDAR_RRPN2polys(image[:-4], bboxes_np, threshold=0.7, result_dir=result_dir, height=height, width=width)

#######################################################################
## clw modify:对于较大的图片,需要先crop,然后得到所有bbox,再做一次nms
w_len = 800
h_len = 800
h_overlap = 256
w_overlap = 256


for image in imlist:
    impath = os.path.join(image_dir, image)
    print('clw: image_name = ', image)
    # print('image:', impath)
    img = cv2.imread(impath)
    cnt += 1
    tic = time.time()

    #################################################3
    box_res_rotate = []
    label_res_rotate = []
    score_res_rotate = []

    imgH = img.shape[0]
    imgW = img.shape[1]

    if imgH < h_len:
        temp = np.zeros([h_len, imgW, 3], np.uint8) #clw modify: 之前是float32,但后面的函数run_on_opencv_image不支持float32类型的img
        temp[0:imgH, :, :] = img
        img = temp
        imgH = h_len

    if imgW < w_len:
        temp = np.zeros([imgH, w_len, 3], np.uint8)
        temp[:, 0:imgW, :] = img
        img = temp
        imgW = w_len

    for hh in range(0, imgH, h_len - h_overlap):
        if imgH - hh - 1 < h_len:
            hh_ = imgH - h_len
        else:
            hh_ = hh
        for ww in range(0, imgW, w_len - w_overlap):
            if imgW - ww - 1 < w_len:
                ww_ = imgW - w_len
            else:
                ww_ = ww
            src_img = img[hh_:(hh_ + h_len), ww_:(ww_ + w_len), :]

            predictions, bounding_boxes = coco_demo.run_on_opencv_image(src_img)

            # 将输出结果转成tensor,维度比如nx5,
            det_boxes_r_ = bounding_boxes.bbox.data.cpu().numpy()
            det_scores_r_ = bounding_boxes.extra_fields['scores'].numpy()
            det_category_r_ = bounding_boxes.extra_fields['labels'].numpy()

            for ii in range(len(det_boxes_r_)):
                box_rotate = det_boxes_r_[ii]
                box_rotate[0] = box_rotate[0] + ww_
                box_rotate[1] = box_rotate[1] + hh_
                box_res_rotate.append(box_rotate)
                label_res_rotate.append(det_category_r_[ii])
                score_res_rotate.append(det_scores_r_[ii])

    box_res_rotate = np.array(box_res_rotate) #  比如(545, 5),
    label_res_rotate = np.array(label_res_rotate) # 比如(545, ) ,内容(6,6,9,9,9..., 9, 9, 6, 6) 代表small-vehicle等index
    score_res_rotate = np.array(score_res_rotate) # 比如(545, ),内容(0.99, 0.98..., 0.54, 0.99, 0.97..)

    box_res_rotate_ = []
    label_res_rotate_ = []
    score_res_rotate_ = []

    r_threshold = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
                   'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.05, 'plane': 0.3,
                   'large-vehicle': 0.1, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
                   'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3  # }
        , 'helipad': 0.1, 'airport': 0.1, 'container-crane': 0.1}  # clw modify

    for sub_class in range(1, 19):  # 18类物体
        index = np.where(label_res_rotate == sub_class)[0] # 比如label_res_rotate有114个small_vehicle,对应sub_class=6,找到这114个对应的索引
        if len(index) == 0:
            continue
        tmp_boxes_r = box_res_rotate[index]
        tmp_label_r = label_res_rotate[index]
        tmp_score_r = score_res_rotate[index]

        tmp_boxes_r = np.array(tmp_boxes_r)
        tmp = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1]) # 比如原来是(114,5),现在变成了(114,6),为的是后面把score加进来
        tmp[:, 0:-1] = tmp_boxes_r
        tmp[:, -1] = np.array(tmp_score_r)

        try:
        ##########################################
        ##### nms_rotate_cpu()函数, 从R2CNN的nms_rotate中借鉴
        # inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r),
        #                             scores=np.array(tmp_score_r),
        #                             iou_threshold=r_threshold[LABEl_NAME_MAP[sub_class]],
        #                             max_output_size=500)
        #########################################################
            boxes = np.array(tmp_boxes_r)
            scores = np.array(tmp_score_r)
            iou_threshold = r_threshold[LABEl_NAME_MAP[sub_class]]
            max_output_size = 500

            keep = []

            order = scores.argsort()[::-1]
            num = boxes.shape[0]

            suppressed = np.zeros((num), dtype=np.int)

            for _i in range(num):
                if len(keep) >= max_output_size:
                    break

                i = order[_i]
                if suppressed[i] == 1:
                    continue
                keep.append(i)
                r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
                area_r1 = boxes[i, 2] * boxes[i, 3]
                for _j in range(_i + 1, num):
                    j = order[_j]
                    if suppressed[i] == 1:
                        continue
                    r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                    area_r2 = boxes[j, 2] * boxes[j, 3]
                    inter = 0.0

                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5) # 防止除以0

                    if inter >= iou_threshold:
                        suppressed[j] = 1

            inx = np.array(keep, np.int64)
        ######################################################
        except:
            # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
            jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
            jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
            inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                 float(r_threshold[LABEl_NAME_MAP[sub_class]]), 0)

        ######################################################
        box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
        score_res_rotate_.extend(np.array(tmp_score_r)[inx])
        label_res_rotate_.extend(np.array(tmp_label_r)[inx])
    ################################################################################


    #########################################
    ##clw modify: 测试使用
    ##import matplotlib.pyplot as plt
    ##plt.imshow(predictions[:, :, [2, 1, 0]])
    ##plt.axis("off")
    ##plt.show()
    #########################################
    toc = time.time()

    print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    box_res_rotate_np = np.array(box_res_rotate_)
    ##box_res_rotate_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN #clw note:如果训练时没有用T.randomRotation,就不会扩边,那这里也不用除

    width, height = bounding_boxes.size

    if vis:
        pil_image = vis_image(Image.fromarray(img), box_res_rotate_np) # clw modify
        pil_image.save(image)
        #pil_image.show()
        # time.sleep(1)  # clw modify: 之前是10


    else:
        #write_result_ICDAR_RRPN2polys(image[:-4], bboxes_np, threshold=0.7, result_dir=result_dir, height=height, width=width)
        # clw modify
        # eval txt
        CLASS_DOTA = NAME_LABEL_MAP.keys()
        # Task1
        write_handle_r = {}

        txt_dir_r = './txt_output'
        if not os.path.exists(txt_dir_r):
            os.mkdir(txt_dir_r)

        for sub_class in CLASS_DOTA:
            if sub_class == 'back_ground':
                continue
            write_handle_r[sub_class] = open(os.path.join(txt_dir_r, '%s.txt' % sub_class), 'a+')

        #######################################################################
        # clw note:把下面这个R2CNN的函数借用过来
        # rboxes = coordinate_convert.forward_convert(box_res_rotate_, with_label=False)

        #def forward_convert(coordinate, with_label=True):
        #coordinate = box_res_rotate_
        coordinate = list(box_res_rotate_np)

        """
        :param coordinate: format [x_c, y_c, w, h, theta]
        :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
        """
        boxes = []

        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), -rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

        rboxes = np.array(boxes, dtype=np.float32)

        #####################################################################

        for i, rbox in enumerate(rboxes):
            command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (impath.split('/')[-1].split('.')[0],
                                                                             score_res_rotate_[i],
                                                                             rbox[0], rbox[1], rbox[2], rbox[3],
                                                                             rbox[4], rbox[5], rbox[6], rbox[7],)

            write_handle_r[LABEl_NAME_MAP[label_res_rotate_[i]]].write(command)

        for sub_class in CLASS_DOTA:
            if sub_class == 'back_ground':
                continue
            write_handle_r[sub_class].close()


'''
if dataset_name == 'IC15':
    zipfilename = os.path.join(result_dir, 'submit_' + config_file.split('/')[-1].split('.')[0] + '_' + cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0] + '.zip')
    if os.path.isfile(zipfilename):
        print('Zip file exists, removing it...')
        os.remove(zipfilename)
    zip_dir(result_dir, zipfilename)
    comm = 'curl -i -F "submissionFile=@' + zipfilename + '" http://127.0.0.1:8080/evaluate'
    # print(comm)
    print(os.popen(comm, 'r'))
else:
    pass
'''
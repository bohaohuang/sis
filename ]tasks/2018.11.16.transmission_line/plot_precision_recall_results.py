import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils


def get_array(line):
    line = line.split('[')[1][:-2]
    line = [float(a.strip()[1:-1]) for a in line.split(',')]
    return line


def parse_results(results):
    p = get_array(results[2])
    r = get_array(results[3])

    return p, r


def voc_ap(rec, prec):
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)  # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


img_dir, task_dir = sis_utils.get_task_img_folder()
model_name = 'faster_rcnn_2018-11-29_20-13-13'
file_name = os.path.join(task_dir, 'results_{}'.format(model_name), 'results.txt')
results = ersa_utils.load_file(file_name)
p_rcnn, r_rcnn = parse_results(results)
ap_rcnn, _, _ = voc_ap(r_rcnn, p_rcnn)

model_name = 'faster_rcnn_2018-11-25_09-14-12'
file_name = os.path.join(task_dir, 'results_{}'.format(model_name), 'results.txt')
results = ersa_utils.load_file(file_name)
p_rcnn_small, r_rcnn_small = parse_results(results)
ap_rcnn_small, _, _ = voc_ap(r_rcnn, p_rcnn)

model_name = 'confmap_uab_UnetCrop_towers_pw5_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
file_name = os.path.join(task_dir, 'results_{}'.format(model_name), 'results.txt')
results = ersa_utils.load_file(file_name)
p_unet, r_unet = parse_results(results)
ap_unet, _, _ = voc_ap(r_unet, p_unet)

plt.plot(r_rcnn, p_rcnn, '-o', label='RCNN (Large) mAP={:.2f}%'.format(ap_rcnn * 100), markersize=6)
plt.plot(r_rcnn_small, p_rcnn_small, '-o', label='RCNN (Small) mAP={:.2f}%'.format(ap_rcnn_small * 100), markersize=6)
plt.plot(r_unet, p_unet, '-o', label='UNet mAP={:.2f}%'.format(ap_unet * 100), markersize=6)
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Overall Performance Comparison')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'overall_performance_comparison.png'))
plt.show()

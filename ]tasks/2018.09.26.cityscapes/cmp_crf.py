import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sis_utils
import ersa_utils


img_dir, task_dir = sis_utils.get_task_img_folder()

orig_dir = r'/home/lab/Documents/bohao/code/third_party/models/research/deeplab/datasets/cityscapes/leftImg8bit/val/frankfurt'
base_dir = r'/home/lab/Documents/bohao/data/deeplab_model/vis/raw_segmentation_results'
post_dir = r'/home/lab/Documents/bohao/data/deeplab_model/post_10_120_5'

orig_files = sorted(glob(os.path.join(orig_dir, '*_leftImg8bit.png')))

for orig_file in orig_files[:10]:
    f_name = os.path.basename(orig_file)[:-16]

    base_file = os.path.join(base_dir, "b'{}'.png".format(f_name))
    post_file = os.path.join(post_dir, "b'{}'.png".format(f_name))

    orig = ersa_utils.load_file(orig_file)
    base = ersa_utils.load_file(base_file)
    post = ersa_utils.load_file(post_file)

    fig = plt.figure(figsize=(6, 10))
    plt.subplot(311)
    plt.imshow(orig.astype(np.uint8))
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(base)
    plt.axis('off')
    plt.subplot(313)
    plt.imshow(post)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '{}_cmp.png'.format(f_name)))
    plt.close(fig)
    # plt.show()

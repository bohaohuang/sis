import os
import imageio
from glob import glob
from tqdm import tqdm
import utils

samples_dir = r'/home/lab/Documents/bohao/code/third_party/DCGAN-tensorflow/inria_sample'
sample_files = glob(os.path.join(samples_dir, '*.png'))
sample_files = sorted(sample_files)
n_num = len(sample_files)

img_dir, task_dir = utils.get_task_img_folder()

images = []
for cnt in tqdm(range(0, n_num, 5)):
    images.append(imageio.imread(sample_files[cnt]))
imageio.mimsave(os.path.join(img_dir, 'celeb_dcgan_repo.gif'), images, duration=0.2)

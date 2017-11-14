import os


size_range = [i for i in range(224, 2800, 128)]
size_range.extend([2800])
for size in size_range:
    print('Evaluating at patch size {}'.format(size))
    cmd = 'python test_acc_vs_dist.py --input-size={}'.format(size)
    os.system(cmd)

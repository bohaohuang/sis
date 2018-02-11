import os

loo_dir_deeplab = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default'
loo_dir_unet = r'/hdd/Results/UnetCrop_inria_aug_leave_0_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/default'

scores = []

for loo_dir in [loo_dir_deeplab, loo_dir_unet]:
    file_name = os.path.join(loo_dir, 'result.txt')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    A_record = 0
    B_record = 0
    austin = lines[:5]
    for item in austin:
        A = int(item.split('(')[1].split(',')[0])
        B = int(item.split(' ')[-1].split(')')[0])
        A_record += A
        B_record += B
    print(A/B)

import matplotlib.pyplot as plt

n_train = 8000
epoch = 100
resolution = 0.3

time = [17*60*60+19*60+20, 21*60*60+24*60+14]

for model in ['unet', 'deeplab']:
    if model == 'unet':
        size = 572
    elif model == 'deeplab':
        size = 321
    else:
        size = 384



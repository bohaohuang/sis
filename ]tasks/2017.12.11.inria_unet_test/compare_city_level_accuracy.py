import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    city_pair = ['kitsap', 'tyrol-w']
    ious = np.zeros((2, 10))
    city_name_list = []

    for cnt, city_name in enumerate(city_pair):
        model_name = 'UnetInria_fr_mean_reduced_appendix_EP-5_LR-0.0001_CT_{}'.format(city_name)

        # load fine tune model
        file_name = os.path.join(task_dir, model_name+'.npy')
        iou = dict(np.load(file_name).tolist())
        for i, (key, val) in enumerate(iou.items()):
            city_name_list.append(key.split('/')[-1].split('.')[0])
            ious[cnt][i] = val

        # load not fine tuned model
        model_name = 'UnetInria_fr_mean_reduced_EP-100_DS-60.0_LR-0.0001'
        file_name = os.path.join(task_dir, model_name + '.npy')
        iou = dict(np.load(file_name).tolist())
        i = 5
        for key, val in iou.items():
            if city_pair[1-cnt] in key:
                ious[cnt][i] = val
                i += 1

    # plot bars
    N = 5
    ind = np.arange(N)

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(121)
    rect1 = ax.bar(ind-0.35/2, ious[0][5:], 0.35, color='g', label='original')
    rect2 = ax.bar(ind+0.35/2, ious[0][:5], 0.35, color='r', label='fine-tune')
    plt.xticks(ind, city_name_list[:5])
    plt.xlabel('Tile Name')
    plt.ylabel('IoU')
    #utils.barplot_autolabel(ax, rect1, margin=0.02)
    #utils.barplot_autolabel(ax, rect2, margin=0.02)
    plt.legend()
    plt.ylim(0, 0.85)
    plt.title('Train {} Test {}'.format(city_pair[0], city_pair[1]))

    ax = plt.subplot(122)
    rect1 = ax.bar(ind - 0.35 / 2, ious[1][5:], 0.35, color='g', label='original')
    rect2 = ax.bar(ind + 0.35 / 2, ious[1][:5], 0.35, color='r', label='fine-tune')
    plt.xticks(ind, city_name_list[5:])
    plt.xlabel('Tile Name')
    plt.ylabel('IoU')
    #utils.barplot_autolabel(ax, rect1, margin=0.02)
    #utils.barplot_autolabel(ax, rect2, margin=0.02)
    plt.legend()
    plt.ylim(0, 0.85)
    plt.title('Train {} Test {}'.format(city_pair[1], city_pair[0]))

    plt.savefig(os.path.join(img_dir, 'finetune_comparison_{}_{}.png'.format(city_pair[0], city_pair[1])))
    plt.show()

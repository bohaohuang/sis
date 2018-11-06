import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils
from ersa_utils import read_tensorboard_csv


img_dir, task_dir = utils.get_task_img_folder()


def plot_old():
    file_name_temp = 'run_unet_aemo_PS(572, 572)_BS5_EP60_LR0.001_DS{}_DR0.1-tag-{}.csv'

    plt.figure(figsize=(10, 6))

    # IoU
    ax1 = plt.subplot(211)
    for ds in [20, 40]:
        file_name = file_name_temp.format(ds, 'IoU')
        step, value = read_tensorboard_csv(os.path.join(task_dir, file_name), 5, 2)

        plt.plot(step, value, label='decay step={}'.format(ds), linewidth=2)
    plt.ylabel('IoU')
    plt.grid(True)
    plt.title('Training Comparison')

    # learning rate
    ax2 = plt.subplot(212, sharex=ax1)
    for ds in [20, 40]:
        file_name = file_name_temp.format(ds, 'learning_rate_1')
        step, value = read_tensorboard_csv(os.path.join(task_dir, file_name), 3, 0)

        plt.plot(step, value, label='decay step={}'.format(ds), linewidth=2)
    plt.legend()
    plt.xlabel('Step Number')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'train_crve_cmp.png'))
    plt.show()


def get_plot_vals(file_name_temp, fold_name='new5', range_len=4):
    val_list = []
    step = []
    for i in range(range_len):
        file_name = os.path.join(task_dir, fold_name, file_name_temp.format(i))
        step, value = read_tensorboard_csv(os.path.join(task_dir, file_name))
        val_list.append(value)
    val_list = np.stack(val_list, axis=0)
    val = np.mean(val_list, axis=0)
    val_min = np.min(val_list, axis=0)
    val_max = np.max(val_list, axis=0)
    return val, val_min, val_max, step


def plot_new5():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp)
    plt.plot(step, val, label='Finetune', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    # scratch lr1e-4
    file_name_temp = 'run_unet_aemo_scratch_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp)
    plt.plot(step, val, label='Scratch 1e-4', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    # scratch lr1e-3
    file_name_temp = 'run_unet_aemo_scratch_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp)
    plt.plot(step, val, label='Scratch 1e-3', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    plt.title('Validation IoU with Tile-wise Hist Matching')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    plt.ylim([0.48, 0.74])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'new5_run_cmp.png'))
    plt.show()


def plot_new4():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new4')
    plt.plot(step, val, '-', label='Finetune 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new4')
    plt.plot(step, val, '--', label='Finetune 1e-4', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    for up in range(6, 10):
        file_name_temp = 'run_unet_aemo_up{}'.format(up) + '_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
        val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new4')
        plt.plot(step, val, '-', label='Clamp{} 1e-3'.format(up), linewidth=2, color=colors[up-5])
        plt.fill_between(step, val_max, val_min, facecolor=colors[up-5], alpha=0.1, interpolate=True)

        file_name_temp = 'run_unet_aemo_up{}'.format(up) + '_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
        val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new4')
        plt.plot(step, val, '--', label='Clamp{} 1e-4'.format(up), linewidth=2, color=colors[up - 5])
        plt.fill_between(step, val_max, val_min, facecolor=colors[up - 5], alpha=0.1, interpolate=True)

    plt.title('Validation IoU with Panel-wise Hist Matching')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    plt.ylim([0.48, 0.74])
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'new4_run_cmp.png'))
    plt.show()


def plot_new3():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Raw Finetune 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '--', label='Raw Finetune 1e-4', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_hist_{}_hist_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Hist Finetune 1e-3', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_hist_{}_hist_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '--', label='Hist Finetune 1e-4', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    for up in range(7, 9):
        file_name_temp = 'run_unet_aemo_hist_up{}'.format(up) + '_{}_hist_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
        val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
        plt.plot(step, val, '-', label='Clamp{} 1e-3'.format(up), linewidth=2, color=colors[up-5])
        plt.fill_between(step, val_max, val_min, facecolor=colors[up-5], alpha=0.1, interpolate=True)

        file_name_temp = 'run_unet_aemo_hist_up{}'.format(up) + '_{}_hist_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
        val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
        plt.plot(step, val, '--', label='Clamp{} 1e-4'.format(up), linewidth=2, color=colors[up - 5])
        plt.fill_between(step, val_max, val_min, facecolor=colors[up - 5], alpha=0.1, interpolate=True)

    plt.title('Validation IoU with Panel-wise Hist Matching')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    plt.ylim([0.48, 0.74])
    plt.legend(loc='upper left', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'new3_run_cmp.png'))
    plt.show()


def plot_old_curves():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.01_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='old')
    plt.plot(step, val, '-', label='Raw Finetune 1e-2', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Raw Finetune 1e-3', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Raw Finetune 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR1e-05_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='old')
    plt.plot(step, val, '-', label='Raw Finetune 1e-5', linewidth=2, color=colors[3])
    plt.fill_between(step, val_max, val_min, facecolor=colors[3], alpha=0.1, interpolate=True)

    plt.title('Validation IoU on Raw Data')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    plt.ylim([0.48, 0.74])
    plt.legend(loc='upper left', ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'old_all_run_cmp.png'))
    plt.show()


def plot_old_cmp_new():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Raw Finetune 1e-3', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Raw Finetune 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_hist_{}_hist_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Hist Finetune 1e-3', linewidth=2, color=colors[4])
    plt.fill_between(step, val_max, val_min, facecolor=colors[4], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_hist_{}_hist_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Hist Finetune 1e-4', linewidth=2, color=colors[5])
    plt.fill_between(step, val_max, val_min, facecolor=colors[5], alpha=0.1, interpolate=True)

    plt.title('Validation IoU Raw Hist Comparison')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    plt.ylim([0.48, 0.74])
    plt.legend(loc='upper left', ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'old_all_run_cmp_new.png'))
    plt.show()


def plot_old_cmp_new_scratch():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Raw Finetune 1e-3', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Raw Finetune 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_hist_{}_hist_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Hist Finetune 1e-3', linewidth=2, color=colors[4])
    plt.fill_between(step, val_max, val_min, facecolor=colors[4], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_hist_{}_hist_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new3')
    plt.plot(step, val, '-', label='Hist Finetune 1e-4', linewidth=2, color=colors[5])
    plt.fill_between(step, val_max, val_min, facecolor=colors[5], alpha=0.1, interpolate=True)

    # scratch
    file_name_temp = 'run_unet_aemo_scratch_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new4')
    plt.plot(step, val, '-.', label='Raw Scratch 1e-3', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_scratch_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='new4')
    plt.plot(step, val, '-.', label='Raw Scratch 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_scratch_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp)
    plt.plot(step, val, '-.', label='Hist Scratch 1e-3', linewidth=2, color=colors[4])
    plt.fill_between(step, val_max, val_min, facecolor=colors[4], alpha=0.1, interpolate=True)

    file_name_temp = 'run_unet_aemo_scratch_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1-tag-IoU.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp)
    plt.plot(step, val, '-.', label='Hist Scratch 1e-4', linewidth=2, color=colors[5])
    plt.fill_between(step, val_max, val_min, facecolor=colors[5], alpha=0.1, interpolate=True)

    plt.title('Validation IoU')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    plt.ylim([0.48, 0.74])
    plt.legend(loc='lower right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'old_all_run_cmp_new_scratch.png'))
    plt.show()


def plot_uab_ft():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '-', label='Hist Finetune 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.0005_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '-', label='Hist Finetune 5e-4', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '-', label='Hist Finetune 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    # scratch files name
    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '--', label='Hist Scratch 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.0005_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '--', label='Hist Scratch 5e-4', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '--', label='Hist Scratch 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    # finetune files name
    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo')
    plt.plot(step, val, '-.', label='Raw Finetune 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.0005_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo')
    plt.plot(step, val, '-.', label='Raw Finetune 5e-4', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo')
    plt.plot(step, val, '-.', label='Raw Finetune 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    # scratch files name
    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo')
    plt.plot(step, val, '--.', label='Raw Scratch 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.0005_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '--.', label='Raw Scratch 5e-4', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.0001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '--.', label='Raw Scratch 1e-4', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    plt.title('Validation IoU')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    plt.ylim([0.5, 0.6])
    plt.legend(loc='lower right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'uab_cmp_all_ylim.png'))
    plt.show()


def plot_uab_ft_up():
    plt.figure(figsize=(10, 6))
    colors = ersa_utils.get_default_colors()

    # finetune files name
    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '-', label='Hist Finetune 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_up3_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist', range_len=2)
    plt.plot(step, val, '-', label='Hist Finetune 1e-3 Clamp 3', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_up5_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist', range_len=2)
    plt.plot(step, val, '-', label='Hist Finetune 1e-3 Clamp 5', linewidth=2, color=colors[3])
    plt.fill_between(step, val_max, val_min, facecolor=colors[3], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_up7_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist', range_len=2)
    plt.plot(step, val, '-', label='Hist Finetune 1e-3 Clamp 7', linewidth=2, color=colors[4])
    plt.fill_between(step, val_max, val_min, facecolor=colors[4], alpha=0.1, interpolate=True)

    # scratch files name
    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo_hist')
    plt.plot(step, val, '--', label='Hist Scratch 1e-3', linewidth=2, color=colors[0])
    plt.fill_between(step, val_max, val_min, facecolor=colors[0], alpha=0.1, interpolate=True)

    # finetune files name
    file_name_temp = 'run_UnetCrop_aemo_ft_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo')
    plt.plot(step, val, '-', label='Raw Finetune 1e-3', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_up3_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo', range_len=2)
    plt.plot(step, val, '--', label='Raw Finetune 1e-3 Clamp 3', linewidth=2, color=colors[2])
    plt.fill_between(step, val_max, val_min, facecolor=colors[2], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_up5_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo', range_len=2)
    plt.plot(step, val, '--', label='Raw Finetune 1e-3 Clamp 5', linewidth=2, color=colors[3])
    plt.fill_between(step, val_max, val_min, facecolor=colors[3], alpha=0.1, interpolate=True)

    file_name_temp = 'run_UnetCrop_aemo_ft_{}_up7_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo', range_len=2)
    plt.plot(step, val, '--', label='Raw Finetune 1e-3 Clamp 7', linewidth=2, color=colors[4])
    plt.fill_between(step, val_max, val_min, facecolor=colors[4], alpha=0.1, interpolate=True)

    # scratch files name
    file_name_temp = 'run_UnetCrop_aemo_sc_{}_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32-tag-iou_validation.csv'
    val, val_min, val_max, step = get_plot_vals(file_name_temp, fold_name='aemo')
    plt.plot(step, val, '--', label='Raw Scratch 1e-3', linewidth=2, color=colors[1])
    plt.fill_between(step, val_max, val_min, facecolor=colors[1], alpha=0.1, interpolate=True)

    plt.title('Validation IoU')
    plt.xlabel('Step')
    plt.ylabel('IoU')
    #plt.ylim([0.5, 0.6])
    plt.legend(loc='lower right', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, 'uab_cmp_all_ylim.png'))
    plt.show()


if __name__ == '__main__':
    plot_uab_ft_up()

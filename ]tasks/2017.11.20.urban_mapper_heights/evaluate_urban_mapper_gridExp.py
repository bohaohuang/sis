import os
import utils
import numpy as np


def evaluate_results(rsr_data_dir, valid_data_dir, input_size,
                     model_name, num_classes, ckdir, city_name,
                     height_mode, batch_size):
    result = utils.test_authentic_unet_height(rsr_data_dir,
                                              valid_data_dir,
                                              input_size,
                                              model_name,
                                              num_classes,
                                              ckdir,
                                              city_name,
                                              batch_size,
                                              ds_name='urban_mapper',
                                              height_mode=height_mode)
    _, task_dir = utils.get_task_img_folder()
    np.save(os.path.join(task_dir, '{}.npy'.format(model_name)), result)

    return result

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # set parameters
    pre_trained_model = r'/home/lab/Documents/bohao/code/sis/test/models/UnetInria_Origin_fr_resample'
    rsr_data_dir = r'/media/ei-edl01/data/remote_sensing_data'
    valid_data_dir = 'dcc_urban_mapper_height_valid'
    input_size = [572, 572]
    num_classes = 2
    ckdir = r'/home/lab/Documents/bohao/code/sis/test/models/UrbanMapper_Height_GridExp'
    city_name = 'JAX,TAM'
    height_mode = 'subtract'
    batch_size = 5

    _, task_dir = utils.get_task_img_folder()

    epochs = 25
    decay_step = 20
    decay_rate = 0.1
    lr_base = 1e-4
    for ly2kp in range(7, 10):
        layers_to_keep_num = [i for i in range(1, ly2kp + 1)]
        # for lr in [0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01]:
        for lr in [0.5]:
            learning_rate = lr * lr_base

            model_name = '{})_rescaled_EP-{}_DS-{}_DR-{}_LY-{}_LR-{}-{:1.1e}'.format(
                pre_trained_model.split('/')[-1],
                epochs,
                decay_step,
                decay_rate,
                ly2kp,
                lr,
                lr_base)

            print('Evaluating model: {}'.format(model_name))
            result = evaluate_results(rsr_data_dir, valid_data_dir, input_size,
                                      model_name, num_classes, ckdir, city_name,
                                      height_mode, batch_size)
            print(result)
            '''iou = []
            for k, v in result.items():
                iou.append(v)
            result_mean = np.mean(iou)
            print('\t Mean IoU on Validation Set: {:.3f}'.format(result_mean))

            with open(os.path.join(task_dir, 'grid_exp_record_2.txt'), 'a') as record_file:
                record_file.write('{} {}\n'.format(model_name, result_mean))'''
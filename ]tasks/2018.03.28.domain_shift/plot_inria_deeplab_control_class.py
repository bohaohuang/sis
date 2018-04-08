import os
import numpy as np
import matplotlib.pyplot as plt
import utils

base_model_dir = r'/hdd/Results/control_city/DeeplabV3_inria_aug_train_austin_chicago_kitsap_tyrol-w_vienna_' \
                 r'PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'
class_control_model_dir = r'/hdd/Results/control_class/DeeplabV3_inria_aug_train_{}_{}_PS(321, 321)_BS5_' \
                          r'EP100_LR1e-05_DS40_DR0.1_SFN32'
city_control_model_dir = r'/hdd/Results/control_city/DeeplabV3_inria_aug_train_{}_PS(321, 321)_BS5_EP100_' \
                         r'LR1e-05_DS40_DR0.1_SFN32'
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
plt_legend = ['LOO', 'LH1', 'LH0', 'Base']
img_dir, task_dir = utils.get_task_img_folder()

base_model_result = os.path.join(base_model_dir, 'inria', 'result.txt')
for city in city_list:
    result_file_list = []
    result_a, result_b = np.zeros((len(plt_legend), len(city_list)+1)), np.zeros((len(plt_legend), len(city_list)+1))
    train_city_str = '_'.join([a for a in city_list if a != city])
    control_city_result = os.path.join(city_control_model_dir.format(train_city_str), 'inria', 'result.txt')
    result_file_list.append(control_city_result)
    for mode in ['h0', 'h1']:
        control_model_result = os.path.join(class_control_model_dir.format(mode, city), 'inria', 'result.txt')
        result_file_list.append(control_model_result)
    result_file_list.append(base_model_result)

    for cnt, file in enumerate(result_file_list):
        if os.path.exists(file):
            with open(file, 'r') as f:
                results = f.readlines()
        else:
            continue
        for line in results[:-1]:
            A, B = line.split('(')[1].strip().strip(')').split(',')
            result_a[cnt][city_dict[line[:3]]] += float(A)
            result_b[cnt][city_dict[line[:3]]] += float(B)
        result_a[cnt][-1] = np.sum(result_a[cnt][:-1])
        result_b[cnt][-1] = np.sum(result_b[cnt][:-1])
    result = result_a/result_b

    plt.figure(figsize=(9, 5))
    plt.rcParams.update({'font.size': 12})
    ind = np.arange(len(city_list)+1)
    width = 0.2
    for i in range(len(plt_legend)):
        plt.bar(ind+width*i, result[i, :], width=width, label=plt_legend[i])
    plt.xticks(ind + width * 1.5, city_list+['Avg'])
    plt.ylim(0.55, 0.82)
    plt.ylabel('IoU')
    plt.title('Performance Comparison (Leave {})'.format(city))
    plt.legend(loc='upper left', ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'cmp_leave_{}.png'.format(city)))
    plt.show()

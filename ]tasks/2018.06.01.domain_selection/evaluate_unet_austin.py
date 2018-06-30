import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]


for runId in [0]:
    for model_dir in [
        #'/hdd6/Models/Inria_Domain/UnetCrop_inria_austin_0_PS(572, 572)_BS5_EP40_LR1e-05_DS20_DR0.1_SFN32',
            #'/hdd6/Models/Inria_Domain/UnetCrop_inria_austin_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
            #r'/hdd6/Models/Inria_Domain/UnetCrop_inria_chicago_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
            #r'/hdd6/Models/Inria_Domain/UnetCrop_inria_kitsap_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
            #r'/hdd6/Models/Inria_Domain/UnetCrop_inria_tyrol-w_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
            #r'/hdd6/Models/Inria_Domain/UnetCrop_inria_vienna_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        #r'/hdd6/Models/Inria_Domain2/UnetCrop_inria_austin_tile2048_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        #r'/hdd6/Models/Inria_Domain2/UnetCrop_inria_chicago_tile2048_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        #r'/hdd6/Models/Inria_Domain2/UnetCrop_inria_kitsap_tile2048_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        #r'/hdd6/Models/Inria_Domain2/UnetCrop_inria_vienna_tile2048_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        r'/hdd6/Models/Inria_Domain_LOO/UnetPredict_inria_loo_mtl_cust_1_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        r'/hdd6/Models/Inria_Domain_LOO/UnetPredict_inria_loo_mtl_cust_2_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        r'/hdd6/Models/Inria_Domain_LOO/UnetPredict_inria_loo_mtl_cust_3_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        r'/hdd6/Models/Inria_Domain_LOO/UnetPredict_inria_loo_mtl_cust_4_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
    ]:

        tf.reset_default_graph()

        blCol = uab_collectionFunctions.uabCollection('inria')
        blCol.readMetadata()
        file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
        file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
        idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
        idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
        # use first 5 tiles for validation
        file_list_valid = uabCrossValMaker.make_file_list_by_key(
            idx, file_list, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
        file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
            idx_truth, file_list_truth, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
        img_mean = blCol.getChannelMeans([0, 1, 2])

        # make the model
        # define place holder
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
        y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                                  trainable=mode,
                                                  input_size=input_size,
                                                  batch_size=5)
        # create graph
        model.create_graph('X', class_num=2)

        # evaluate on tiles
        model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                       input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                       save_result_parent_dir='domain_selection', ds_name='inria',
                       best_model=True)

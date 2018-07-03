import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import keras
import numpy as np
import utils
import uabRepoPaths
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions
import building_data_reader


if __name__ == '__main__':
    # train on um
    blCol_um = uab_collectionFunctions.uabCollection('um')
    img_mean_um = blCol_um.getChannelMeans([0, 1, 2])
    blCol_um.readMetadata()
    # valid on inria
    blCol_inria = uab_collectionFunctions.uabCollection('inria')
    img_mean_inria = blCol_inria.getChannelMeans([0, 1, 2])
    blCol_inria.readMetadata()

    model_name = 'unet'
    batch_size = 25
    class_num = 2
    n_train = 300
    n_valid = 100
    epoch = 100
    learn_rate = 1e-5
    prescr_name = 'incep'
    img_dir, task_dir = utils.get_task_img_folder()

    if prescr_name == 'incep':
        center_crop = (299, 299)
        prescr = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet')
        avg_pool = prescr.get_layer('avg_pool').output
        flat = keras.layers.Dropout(0.5)(avg_pool)
        x = keras.layers.Dense(100, activation='relu')(avg_pool)
        predictions = keras.layers.Dense(class_num, activation='softmax')(x)
    elif prescr_name == 'res50':
        center_crop = (224, 224)
        prescr = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
        flat = prescr.get_layer('flatten_1').output
        flat = keras.layers.Dropout(0.5)(flat)
        x = keras.layers.Dense(100, activation='relu')(flat)
        predictions = keras.layers.Dense(class_num, activation='softmax')(x)
    else:
        raise KeyError
    model = keras.Model(inputs=prescr.input, outputs=predictions)
    '''for layer in prescr.layers:
        layer.trainable = False'''

    if model_name == 'unet':
        patch_size = (572, 572)
        overlap = 184
        pad = 92
    else:
        patch_size = (321, 321)
        overlap = 0
        pad = 0

    # extract patches
    extrObj_um = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                       cSize=patch_size,
                                                       numPixOverlap=overlap,
                                                       extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                       isTrain=True,
                                                       gtInd=3,
                                                       pad=pad)
    patchDir_um = extrObj_um.run(blCol_um)
    extrObj_inria = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                          cSize=patch_size,
                                                          numPixOverlap=overlap,
                                                          extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                          isTrain=True,
                                                          gtInd=3,
                                                          pad=pad)
    patchDir_inria = extrObj_inria.run(blCol_inria)

    for leave_city in range(1):
        # make data reader
        chipFiles = os.path.join(patchDir_inria, 'fileList.txt')
        idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir_inria, 'fileList.txt', 'force_tile')
        file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

        idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir_um, 'fileList.txt', 'force_tile')
        file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(20, 136)])

        dataReader_train = building_data_reader.ImageLabelReaderBuilding(
            [3], [0, 1, 2], patchDir_um, file_list_train, patch_size, batch_size, center_crop, 0.1,
            block_mean=np.append([0], img_mean_um), dataAug='flip,rotate').readManager
        dataReader_valid = building_data_reader.ImageLabelReaderBuilding(
            [3], [0, 1, 2], patchDir_inria, file_list_valid, patch_size, batch_size, center_crop, 0.1,
            block_mean=np.append([0], img_mean_inria)).readManager

        model_dir = os.path.join(uabRepoPaths.modelPath, 'Prescreen', prescr_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_save_dir = os.path.join(model_dir, '{}_building_loo_{}.hdf5'.format(prescr_name, leave_city))
        ckpt = keras.callbacks.ModelCheckpoint(model_save_dir, monitor='categorical_crossentropy', verbose=0)
        tb = keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'), histogram_freq=0, write_graph=True,
                                         write_images=True)
        optm = keras.optimizers.Adam(lr=learn_rate, decay=0.1)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit_generator(dataReader_train, steps_per_epoch=n_train, epochs=epoch, validation_data=dataReader_valid,
                            validation_steps=n_valid, callbacks=[ckpt, tb], verbose=2)

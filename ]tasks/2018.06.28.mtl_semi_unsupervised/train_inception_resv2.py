import os
import keras
import numpy as np
import sis_utils
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions
import building_data_reader


if __name__ == '__main__':
    blCol = uab_collectionFunctions.uabCollection('inria')
    img_mean = blCol.getChannelMeans([0, 1, 2])
    model_name = 'unet'
    batch_size = 25
    class_num = 2
    n_train = 400
    n_valid = 10
    epoch = 10
    learn_rate = 1e-5
    prescr_name = 'incep'
    img_dir, task_dir = sis_utils.get_task_img_folder()

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
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=patch_size,
                                                    numPixOverlap=overlap,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=pad)
    patchDir = extrObj.run(blCol)

    for leave_city in range(1):
        # make data reader
        chipFiles = os.path.join(patchDir, 'fileList.txt')
        idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'city')
        idx2, _ = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
        idx3 = [j * 10 + i for i, j in zip(idx, idx2)]
        filter_train = []
        filter_valid = []
        for i in range(5):
            for j in range(1, 37):
                if i == leave_city and j <= 5:
                    filter_valid.append(j * 10 + i)
                elif i != leave_city:
                    filter_train.append(j * 10 + i)
        file_list_train = uabCrossValMaker.make_file_list_by_key(idx3, file_list, filter_train)
        file_list_valid = uabCrossValMaker.make_file_list_by_key(idx3, file_list, filter_valid)

        dataReader_train = building_data_reader.ImageLabelReaderBuilding(
            [3], [0, 1, 2], patchDir, file_list_train, patch_size, batch_size, center_crop, 0.1,
            block_mean=np.append([0], img_mean), dataAug='flip,rotate').readManager
        dataReader_valid = building_data_reader.ImageLabelReaderBuilding(
            [3], [0, 1, 2], patchDir, file_list_valid, patch_size, batch_size, center_crop, 0.1,
            block_mean=np.append([0], img_mean)).readManager

        model_save_dir = os.path.join(task_dir, '{}_building_loo_{}.hdf5'.format(prescr_name, leave_city))
        ckpt = keras.callbacks.ModelCheckpoint(model_save_dir, monitor='categorical_crossentropy', verbose=1)
        optm = keras.optimizers.Adam(lr=learn_rate)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit_generator(dataReader_train, steps_per_epoch=n_train, epochs=epoch, validation_data=dataReader_valid,
                            validation_steps=n_valid, callbacks=[ckpt], verbose=2)

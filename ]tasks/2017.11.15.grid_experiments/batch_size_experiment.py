import argparse
import utils

TEST_DATA_DIR = 'dcc_inria_valid'
CITY_NAME = 'austin'
RSR_DATA_DIR = r'/media/ei-edl01/data/remote_sensing_data'
PATCH_DIR = r'/media/ei-edl01/user/bh163/data/iai'
TEST_PATCH_APPENDIX = 'valid_noaug_dcc'
TEST_TILE_NAMES = ','.join(['{}'.format(i) for i in range(1, 6)])
RANDOM_SEED = 1234
BATCH_SIZE = 2
INPUT_SIZE = 224
CKDIR = r'/home/lab/Documents/bohao/code/sis/test/models/GridExp'
MODEL_NAME = 'UNET_austin_no_random'
NUM_CLASS = 2
GPU = '0'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-dir', default=TEST_DATA_DIR, help='path to release folder')
    parser.add_argument('--rsr-data-dir', default=RSR_DATA_DIR, help='path to rsrClassData folder')
    parser.add_argument('--patch-dir', default=PATCH_DIR, help='path to patch directory')
    parser.add_argument('--test-patch-appendix', default=TEST_PATCH_APPENDIX, help='valid patch appendix')
    parser.add_argument('--test-tile-names', default=TEST_TILE_NAMES, help='image tiles')
    parser.add_argument('--city-name', type=str, default=CITY_NAME, help='city name (default austin)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED, help='tf random seed')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (10)')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, help='input size 224')
    parser.add_argument('--ckdir', default=CKDIR, help='ckpt dir (models)')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASS, help='# classes (including background)')
    parser.add_argument('--GPU', type=str, default=GPU, help="GPU used for computation.")
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Model name')

    flags = parser.parse_args()
    flags.input_size = (flags.input_size, flags.input_size)
    return flags


def main(flags):

    for batch_size in [1, 2, 5, 10]:
        model_name = 'UNET_PS-{}__BS-{}__E-100__NT-8000__DS-60__CT-__no_random'.format(flags.input_size, batch_size)
        print(model_name)

        result = utils.test_unet(flags.rsr_data_dir,
                                 flags.test_data_dir,
                                 flags.input_size,
                                 model_name,
                                 flags.num_classes,
                                 flags.ckdir,
                                 flags.city_name,
                                 flags.batch_size)
        print(result)


if __name__ == '__main__':
    flags = read_flag()
    main(flags)
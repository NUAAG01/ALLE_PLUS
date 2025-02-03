from mini_batch_loader import *
import sys
import time
import State
import os
import torch
import pixelwise_a3c_el
import MyFCN_el
import torch.nn as nn
import copy
import pandas as pd

# _/_/_/ paths _/_/_/

TRAINING_DATA_PATH = "D:/Image_Enhancement_code/ALL-E/ReLLIE/data/label.txt"
TESTING_DATA_PATH = "D:\Image_Enhancement_code\ALL-E\ALL-E_1/net.txt"
IMAGE_DIR_PATH = "D:\Image_Enhancement_code\ALL-E\ALL-E_1/"
SAVE_PATH = "D:/Image_Enhancement_code/ALL-E/ReLLIE/model_10/fpop_myfcn_"

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 1  # must be 1
N_EPISODES = 3000
EPISODE_LEN = 6
SNAPSHOT_EPISODES = 3000
TEST_EPISODES = 3000
GAMMA = 1.05  # discount factor

# noise setting


N_ACTIONS = 27
MOVE_RANGE = 27  # number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 224
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
GPU_ID = 0


def test(loader1, agent_el, fout):
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)
    cnt = 1
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        print(cnt)
        cnt = cnt + 1
        raw_x = loader1.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))

        current_state.reset(raw_x)

        for t in range(0, EPISODE_LEN):

            action_el = agent_el.act(current_state.image)
            current_state.step(action_el)

        agent_el.stop_episode()

        p = np.maximum(0, current_state.image)
        p = np.minimum(1, p)
        p = (p * 255).astype(np.uint8)
        p = np.squeeze(p, axis=0)
        p = np.transpose(p, (1, 2, 0))

        cv2.imwrite('./result/' + str(i + 1) + '.png', p)

    sys.stdout.flush()


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        # TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)


    pixelwise_a3c_el.chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)

    # load myfcn model
    model_el = MyFCN_el.MyFcn(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer_el = pixelwise_a3c_el.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer_el.setup(model_el)

    agent_el = pixelwise_a3c_el.PixelWiseA3C(model_el, optimizer_el, EPISODE_LEN, GAMMA)
    pixelwise_a3c_el.chainer.serializers.load_npz('model_last/fpop_myfcn_2000/model.npz', agent_el.model)
    agent_el.act_deterministically = True
    agent_el.model.to_gpu()


    start = time.perf_counter()

    test(mini_batch_loader,agent_el,fout )
    end = time.perf_counter()

    print('Running time: %s Seconds' % (end - start))


if __name__ == '__main__':
    fout = open('testlog9.txt', "w")
    start = time.time()
    main(fout)
    end = time.time()
    print("{s}[s]".format(s=end - start))
    print("{s}[m]".format(s=(end - start) / 60))
    print("{s}[h]".format(s=(end - start) / 60 / 60))
    fout.write("{s}[s]\n".format(s=end - start))
    fout.write("{s}[m]\n".format(s=(end - start) / 60))
    fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
    fout.close()
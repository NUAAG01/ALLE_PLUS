from mini_batch_loader import *
import sys
import time
import State
import State_d
import os
import torch
import pixelwise_a3c_el
import pixelwise_a3c_d
import MyFCN_el
import MyFCN_d
import torch.nn as nn
from thop import profile
import copy
import pandas as pd


# _/_/_/ paths _/_/_/

TRAINING_DATA_PATH = "D:/Image_Enhancement_code/ALL-E/ReLLIE/data/label.txt"
TESTING_DATA_PATH = "D:/Image_Enhancement_code/ALL-E/ReLLIE/eval15/label_low.txt"
IMAGE_DIR_PATH = "D:/Image_Enhancement_code/ALL-E/ReLLIE/eval15/"
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
DENOISE_LEN = 2

# noise setting


N_ACTIONS = 27
MOVE_RANGE = 27  # number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 224
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
GPU_ID = 0


def getimg(img_r, img_g, img_b):
    p_r = np.maximum(0, img_r)
    p_r = np.minimum(1, p_r)
    p_r = (p_r[0] * 255 + 0.5).astype(np.uint8)
    p_r = np.transpose(p_r, (1, 2, 0))
    p_g = np.maximum(0, img_g)
    p_g = np.minimum(1, p_g)
    p_g = (p_g[0] * 255 + 0.5).astype(np.uint8)
    p_g = np.transpose(p_g, (1, 2, 0))
    p_b = np.maximum(0, img_b)
    p_b = np.minimum(1, p_b)
    p_b = (p_b[0] * 255 + 0.5).astype(np.uint8)
    p_b = np.transpose(p_b, (1, 2, 0))
    p1 = np.squeeze(p_r, 2)
    p2 = np.squeeze(p_g, 2)
    p3 = np.squeeze(p_b, 2)
    p = cv2.merge([p3, p2, p1])
    return p


def de(loader, current_state_r, current_state_g, current_state_b, agent, img):
    # len = 2
    raw_x_b, raw_x_g, raw_x_r = loader.data_denoise(img)
    current_state_r.reset(raw_x_r)
    current_state_g.reset(raw_x_g)
    current_state_b.reset(raw_x_b)
    for t in range(0, DENOISE_LEN):
        action_r, inner_state_r = agent.act(current_state_r.tensor)
        action_g, inner_state_g = agent.act(current_state_g.tensor)
        action_b, inner_state_b = agent.act(current_state_b.tensor)
        current_state_r.step(action_r, inner_state_r)
        current_state_g.step(action_g, inner_state_g)
        current_state_b.step(action_b, inner_state_b)
    agent.stop_episode()
    P = getimg(current_state_r.image, current_state_g.image, current_state_b.image)

    return P


def test(loader1, agent_el, agent_d, fout):
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)
    current_state_r = State_d.State_d((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), 1)
    current_state_g = State_d.State_d((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), 1)
    current_state_b = State_d.State_d((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), 1)
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

        p_d = de(loader1, current_state_r,current_state_g,current_state_b, agent_d, p)



        cv2.imwrite('./result/' + str(i + 1) + '.png', p_d)

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
    pixelwise_a3c_el.chainer.serializers.load_npz('./last_model/nima.npz', agent_el.model) 
    agent_el.act_deterministically = True
    agent_el.model.to_gpu()


    ################################
    model_d = MyFCN_d.MyFcn_d(9)
    # _/_/_/ setup _/_/_/
    optimizer = pixelwise_a3c_d.chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model_d)
    agent_d = pixelwise_a3c_d.PixelWiseA3C_InnerState_ConvR(model_d, optimizer, DENOISE_LEN, GAMMA)
    pixelwise_a3c_d.chainer.serializers.load_npz('denoise_model/d_model.npz', agent_d.model)
    agent_d.act_deterministically = True
    agent_d.model.to_gpu()




    start = time.perf_counter()
    test(mini_batch_loader,agent_el, agent_d, fout )
    end = time.perf_counter()

    print('Running time: %s Seconds' % (end - start))


if __name__ == '__main__':
    fout = open('testlog.txt', "w")
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
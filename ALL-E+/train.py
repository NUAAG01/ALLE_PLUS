from mini_batch_loader import *
import sys
import time
import State
import os
import Myloss
import pixelwise_a3c_el
import MyFCN_el
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import copy
from model.model import *


TRAINING_DATA_PATH = "D:/Image_Enhancement_code/ALL-E/ReLLIE/data/label.txt"
IMAGE_DIR_PATH = "D:/Image_Enhancement_code/ALL-E/ReLLIE/our485/low/"
SAVE_PATH = "./model0/fpop_myfcn_"

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 1  # must be 1
N_EPISODES = 3000
EPISODE_LEN = 6
SNAPSHOT_EPISODES = 100
TEST_EPISODES = 100
GAMMA = 1.05  # discount factor

# noise setting


N_ACTIONS = 27
MOVE_RANGE = 27  # number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 224
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
GPU_ID = 0



def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TRAINING_DATA_PATH,
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
    agent_el.model.to_gpu()

    # NIMA model
    base_model = models.vgg16(pretrained=True)
    NIMA_model = NIMA(base_model)
    NIMA_model.load_state_dict(torch.load("premodel/epoch.pth"))
    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NIMA_model = NIMA_model.to(device)

    NIMA_model.eval()
    test_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # _/_/_/ training _/_/_/

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_TV = Myloss.L_TV()
    L_exp = Myloss.L_exp(16, 0.6)
    L_color_rate = Myloss.L_color_rate()
    for episode in range(1, N_EPISODES + 1):
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        r = indices[i:i + TRAIN_BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)
        current_state.reset(raw_x)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        action_value = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        premean = 0.0

        for t in range(0, EPISODE_LEN):
            raw_tensor = torch.from_numpy(raw_x).cuda()
            previous_image = current_state.image.copy()
            action_el = agent_el.act_and_train(current_state.image, reward)
            action_value = (action_el - 9) / 18
            current_state.step(action_el)

            currentImg1 = copy.copy(current_state.image[0, ::])
            currentImg2 = copy.copy(current_state.image[1, ::])
            currentImg1 = np.transpose(currentImg1, (1, 2, 0))
            currentImg2 = np.transpose(currentImg2, (1, 2, 0))
            currentImg1 -= [0.485, 0.456, 0.406]
            currentImg1 /= [0.229, 0.224, 0.225]
            currentImg2 -= [0.485, 0.456, 0.406]
            currentImg2 /= [0.229, 0.224, 0.225]
            currentImg1 = np.transpose(currentImg1, (2, 0, 1))
            currentImg2 = np.transpose(currentImg2, (2, 0, 1))
            currentImg1 = torch.tensor(currentImg1)
            currentImg2 = torch.tensor(currentImg2)
            currentImg1 = currentImg1.unsqueeze(dim=0)
            currentImg2 = currentImg2.unsqueeze(dim=0)
            imt = torch.cat((currentImg1, currentImg2), 0)
            imt = imt.to(device)
            mean = 0.0
            with torch.no_grad():
                out = NIMA_model(imt)
            out = out.view(20, 1)
            for j, e in enumerate(out, 1):
                if j % 10 == 0:
                    mean += 10 * e
                else:
                    mean += (j % 10) * e

            previous_image_tensor = torch.from_numpy(previous_image).cuda()
            current_state_tensor = torch.from_numpy(current_state.image).cuda()
            action_tensor = torch.from_numpy(action_value).cuda()
            loss_spa_cur = torch.mean(L_spa(current_state_tensor, raw_tensor))
            loss_col_cur = 50 * torch.mean(L_color(current_state_tensor))
            Loss_TV_cur = 200 * L_TV(action_tensor)
            A = L_exp(current_state_tensor)
            loss_exp_cur = 80 * torch.mean(L_exp(current_state_tensor))
            loss_col_rate_pre = 20 * torch.mean(L_color_rate(previous_image_tensor, current_state_tensor))
            # reward_previous = loss_spa_pre + loss_col_pre + loss_exp_pre + Loss_TV_pre + loss_col_rate_pre
            # reward_current = loss_spa_cur + loss_exp_cur + Loss_TV_cur + loss_col_rate_pre - 2 * (mean - premean)
            r_aes = 2 * (mean - premean)
            reward_current = loss_spa_cur + loss_exp_cur + Loss_TV_cur + loss_col_rate_pre - r_aes
            premean = mean
            reward = - reward_current
            reward = reward.cpu().numpy()
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        agent_el.stop_episode_and_train(current_state.image, reward, True)

        print("train total reward {a}".format(a=sum_reward))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        if episode % SNAPSHOT_EPISODES == 0:
            agent_el.save(SAVE_PATH + str(episode))

        if i + TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:
            i += TRAIN_BATCH_SIZE

        if i + 2 * TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        


if __name__ == '__main__':
    try:
        fout = open('log.txt', "w")
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
    except Exception as error:
        print(error.message)

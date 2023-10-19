import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *
from skimage.metrics import peak_signal_noise_ratio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/DnCNN-S-50", help='path of log files')
parser.add_argument("--test_data", type=str, default='FID300', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.jpg')) # 테스트하려는 이미지 포맷으로 수정 
    files_source.sort()
    
    
    # process data
    niqe_test_before = 0
    niqe_test_after = 0

    before_folder = os.path.join('data', 'BEFORE')
    output_folder = os.path.join('data', 'AFTER')
    
    
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)

        ISource = torch.Tensor(Img) # 원본 이미지
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise # 노이즈가 추가된 이미지
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
     

        # output = Out.cpu().numpy()

        # 노이즈 추가된 이미지를 BEFORE 폴더에 저장
        before_filename = os.path.join(before_folder, os.path.basename(f))
        before_image = (INoisy.cpu().numpy()[0,0]*255).astype(np.uint8)
        cv2.imwrite(before_filename, before_image)
        
        # 노이즈 제거된 이미지를 AFTER 폴더에 저장
        output_filename = os.path.join(output_folder, os.path.basename(f))
        output_image = (Out.cpu().numpy()[0, 0] * 255).astype(np.uint8)
        cv2.imwrite(output_filename, output_image)
        print(f'Result image saved to {output_filename}')

        # niqe 점수 계산
        niqe_score_before = niqe(before_image)
        niqe_score_after = niqe(output_image)
        
        niqe_test_before += niqe_score_before
        niqe_test_after += niqe_score_after
        print(f'노이즈 제거 전 NIQE: {niqe_score_before:.3f}')
        print(f'노이즈 제거 후 NIQE: {niqe_score_after:.3f}')

    
    niqe_test_before /= len(files_source)
    niqe_test_after /= len(files_source)
    print("\n제거 전 NIQE 평균 %f" % niqe_test_before)
    print("\n제거 후 NIQE 평균 %f" % niqe_test_after)



if __name__ == "__main__":
    main()

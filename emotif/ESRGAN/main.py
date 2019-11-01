import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

def SuperRes(path):
  model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
  print('device loading')
  device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
  print('device loaded')
  model = arch.RRDBNet(3, 3, 64, 23, gc=32)
  model.load_state_dict(torch.load(model_path), strict=True)
  model.eval()
  model = model.to(device)

  print('Model path {:s}. \nTesting...'.format(model_path))

  img = cv2.imread(path, cv2.IMREAD_COLOR)
  print(img.shape)
  img = img * 1.0 / 255
  img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
  img_LR = img.unsqueeze(0)
  img_LR = img_LR.to(device)

  with torch.no_grad():
      output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
  output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
  output = (output * 255.0).round()
  print("writing result")
  cv2.imwrite('ESRGAN/LR/result.png', output)
  print(output.shape)
  return output
SuperRes('ESRGAN/LR/baboon.png')
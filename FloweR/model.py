import torch
import torch.nn as nn
import torch.functional as F

# Define the model
class FloweR(nn.Module):
  def __init__(self, input_size = (384, 384), window_size = 4):
    super(FloweR, self).__init__()

    self.input_size = input_size
    self.window_size = window_size

    #INPUT: 384 x 384 x 10 * 3 

    ### DOWNSCALE ###
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(3 * self.window_size, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 384 x 384 x 128

    self.conv_block_2 = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 192 x 192 x 128

    self.conv_block_3 = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 96 x 96 x 128

    self.conv_block_4 = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 48 x 48 x 128

    self.conv_block_5 = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 24 x 24 x 128

    self.conv_block_6 = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 12 x 12 x 128

    self.conv_block_7 = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 6 x 6 x 128

    self.conv_block_8 = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 3 x 3 x 128

    ### UPSCALE ###
    self.conv_block_9 = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 6 x 6 x 128

    self.conv_block_10 = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 12 x 12 x 128

    self.conv_block_11 = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 24 x 24 x 128

    self.conv_block_12 = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 48 x 48 x 128

    self.conv_block_13 = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 96 x 96 x 128

    self.conv_block_14 = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 192 x 192 x 128
    
    self.conv_block_15 = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
      nn.ReLU(),
    ) # 384 x 384 x 128

    self.conv_block_16 = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding='same')

  def forward(self, x):
    if x.size(1) != self.window_size: 
      raise Exception(f'Shape of the input is not compatable. There should be exactly {self.window_size} frames in an input video.')

    # batch, frames, height, width, colors
    in_x = x.permute((0, 1, 4, 2, 3))
    # batch, frames, colors, height, width

    in_x = in_x.reshape(-1, self.window_size * 3, self.input_size[0], self.input_size[1])

    ### DOWNSCALE ###
    block_1_out = self.conv_block_1(in_x)        # 384 x 384 x 128
    block_2_out = self.conv_block_2(block_1_out) # 192 x 192 x 128
    block_3_out = self.conv_block_3(block_2_out) # 96 x 96 x 128
    block_4_out = self.conv_block_4(block_3_out) # 48 x 48 x 128
    block_5_out = self.conv_block_5(block_4_out) # 24 x 24 x 128
    block_6_out = self.conv_block_6(block_5_out) # 12 x 12 x 128
    block_7_out = self.conv_block_7(block_6_out) # 6 x 6 x 128
    block_8_out = self.conv_block_8(block_7_out) # 3 x 3 x 128

    ### UPSCALE ###
    block_9_out = block_7_out + self.conv_block_9(block_8_out)    # 6 x 6 x 128
    block_10_out = block_6_out + self.conv_block_10(block_9_out)  # 12 x 12 x 128
    block_11_out = block_5_out + self.conv_block_11(block_10_out) # 24 x 24 x 128
    block_12_out = block_4_out + self.conv_block_12(block_11_out) # 48 x 48 x 128
    block_13_out = block_3_out + self.conv_block_13(block_12_out) # 96 x 96 x 128
    block_14_out = block_2_out + self.conv_block_14(block_13_out) # 192 x 192 x 128
    block_15_out = block_1_out + self.conv_block_15(block_14_out) # 384 x 384 x 128

    block_16_out = self.conv_block_16(block_15_out) # 384 x 384 x (2 + 1)
    out = block_16_out.reshape(-1, 3, self.input_size[0], self.input_size[1])

    # batch, colors, height, width
    out = out.permute((0, 2, 3, 1))
    # batch, height, width, colors
    return out
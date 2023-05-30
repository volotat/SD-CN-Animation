import torch
import torch.nn as nn
import torch.functional as F

# Define the model
class FloweR(nn.Module):
  def __init__(self, input_size = (384, 384), window_size = 4):
    super(FloweR, self).__init__()

    self.input_size = input_size
    self.window_size = window_size

    # 2 channels for optical flow
    # 1 channel for occlusion mask
    # 3 channels for next frame prediction 
    self.out_channels = 6
    

    #INPUT: 384 x 384 x 4 * 3 

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
    ) # 3 x 3 x 128 - 9 input tokens

    ### Transformer part ###
    # To be done

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

    self.conv_block_16 = nn.Conv2d(128, self.out_channels, kernel_size=3, stride=1, padding='same')

  def forward(self, input_frames):

    if input_frames.size(1) != self.window_size: 
      raise Exception(f'Shape of the input is not compatable. There should be exactly {self.window_size} frames in an input video.')

    h, w = self.input_size
    # batch, frames, height, width, colors
    input_frames_permuted = input_frames.permute((0, 1, 4, 2, 3))
    # batch, frames, colors, height, width

    in_x = input_frames_permuted.reshape(-1, self.window_size * 3, self.input_size[0], self.input_size[1])

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

    block_16_out = self.conv_block_16(block_15_out) # 384 x 384 x (2 + 1 + 3)
    out = block_16_out.reshape(-1, self.out_channels, self.input_size[0], self.input_size[1])

    ### for future model training ###
    device = out.get_device()

    pred_flow = out[:,:2,:,:] * 255  # (-255, 255)
    pred_occl = (out[:,2:3,:,:] + 1) / 2 # [0, 1]
    pred_next = out[:,3:6,:,:]

    # Generate sampling grids

    # Create grid to upsample input
    '''    
    d = torch.linspace(-1, 1, 8)
    meshx, meshy = torch.meshgrid((d, d))
    grid = torch.stack((meshy, meshx), 2)
    grid = grid.unsqueeze(0) '''
    
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    flow_grid = torch.stack((grid_x, grid_y), dim=0).float()
    flow_grid = flow_grid.unsqueeze(0).to(device=device)
    flow_grid = flow_grid + pred_flow

    flow_grid[:, 0, :, :] = 2 * flow_grid[:, 0, :, :] / (w - 1) - 1
    flow_grid[:, 1, :, :] = 2 * flow_grid[:, 1, :, :] / (h - 1) - 1
    # batch, flow_chanels, height, width
    flow_grid = flow_grid.permute(0, 2, 3, 1)
    # batch, height, width, flow_chanels

    previous_frame = input_frames_permuted[:, -1, :, :, :]
    sampling_mode = "bilinear" if self.training else "nearest" 
    warped_frame = torch.nn.functional.grid_sample(previous_frame, flow_grid, mode=sampling_mode, padding_mode="reflection", align_corners=False)
    alpha_mask = torch.clip(pred_occl * 10, 0, 1) * 0.04
    pred_next = torch.clip(pred_next, -1, 1)
    warped_frame = torch.clip(warped_frame, -1, 1)
    next_frame = pred_next * alpha_mask + warped_frame * (1 - alpha_mask)

    res = torch.cat((pred_flow / 255, pred_occl * 2 - 1, next_frame), dim=1)

    # batch, channels, height, width
    res = res.permute((0, 2, 3, 1))
    # batch, height, width, channels
    return res
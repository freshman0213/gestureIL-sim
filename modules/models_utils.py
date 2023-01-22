import numpy as np

import torch
from torch.distributions import Normal
import torch.nn as nn
from torch.nn import functional as F


def replace_ood(z_mult, z_mult_choices, ood_preds, ood_preds_override, ood_stats, l_mods, device):
    # Corrected Variants
    keys = ["img", "depth", "frc"]
    z_mult_corrected = z_mult.clone()
    # z_mult_choices = [z_mult_noi, z_mult_nod, z_mult_nof]
    replace_key = []
    for idx2 in range(len(ood_preds['img'])):

        # If given ground truth to predict with
        if ood_preds_override is not None:
            replace_idx = ood_preds_override
            z_mult_corrected[idx2] = z_mult_choices[replace_idx][idx2]
            replace_key.append(keys[replace_idx])
            continue

        ood_options = []
        # Get the modalities that are OOD
        for idx, key in enumerate(keys):
            if ood_preds[key][idx2] > 0:
                ood_options.append(idx)

        assert np.sum([ood_preds[xxxx][idx2].item() for xxxx in keys]) == len(ood_options)

        if len(ood_options) > 0:
            mod_errs = []

            for xx in ood_options:
                if keys[xx] == 'frc':
                    num_std_away = l_mods[keys[xx]][idx2] - torch.from_numpy(ood_stats["{}_mse".format(keys[xx])]).to(device)
                    num_std_away /= torch.from_numpy(ood_stats["{}_std".format(keys[xx])]).to(device)
                    num_std_away = num_std_away.min()

                else:
                    num_std_away = l_mods[keys[xx]][idx2] - ood_stats["{}_mse".format(keys[xx])]
                    num_std_away /= ood_stats["{}_std".format(keys[xx])]

                mod_errs.append(num_std_away)

            replace_idx = ood_options[np.argmax(mod_errs)]
            z_mult_corrected[idx2] = z_mult_choices[replace_idx][idx2]
            replace_key.append(keys[replace_idx])
        else:
            replace_key.append("None")

    return replace_key, z_mult_corrected


def predict_ood(inputs, recon_out_all, ood_stats, bs, device):
    (vis_in, frc_in, proprio_in, depth_in, _, movement_mask) = inputs

    # Tensor of shape (bs) where bs is the loss
    (img_recon, img_mask, depth_recon, depth_mask, _, frc_recon, frc_last_recon) = recon_out_all
    l_img = ((img_recon.reshape(bs, -1) - vis_in.reshape(bs, -1)) ** 2).mean(dim=1)
    l_depth = ((depth_recon.reshape(bs, -1) - depth_in.reshape(bs, -1)) ** 2).mean(dim=1)

    frc_channels = frc_in.size()[-1]
    # todo: all force inputs/outputs should be the same size!

    # assert frc_channels == 3 or frc_channels == 6
    l_frc = ((frc_recon.reshape(bs, -1, frc_channels) - frc_in.reshape(bs, -1, frc_channels)) ** 2).mean(dim=1)

    l_mods = {}
    l_mods["img"] = l_img
    l_mods["depth"] = l_depth
    l_mods["frc"] = l_frc

    # Simple OOD threshold check
    keys = ["img", "depth"]
    ood_preds = {}
    for idx, key in enumerate(keys):
        ood_preds[key] = l_mods[key] > ood_stats["{}_thresh".format(key)]

    # Handle special case for frc
    ood_preds["frc"] = l_mods["frc"] > torch.from_numpy(
        ood_stats["{}_thresh".format("frc")]
    ).to(device)
    # print('force shape: ', ood_preds['frc'].shape)
    ood_preds["frc"] = torch.sum(ood_preds["frc"], dim=1).float() > 2.0


    # #just testing
    # ood_preds["frc"] *=0.0
    # ood_preds["img"] *=0.0
    # ood_preds["depth"] = 1
    # print("ood_preds: ", ood_preds)

    return l_mods, ood_preds


def get_basic_inputs(sample_batched, use_cuda, device, n_time_steps=1, str=""):
    if n_time_steps == 1:
        image, force, proprio, depth = (
            sample_batched[str + "image"],
            sample_batched[str + "force"],
            sample_batched[str + "proprio"],
            sample_batched[str + "depth"],
        )
        action = sample_batched["action"]
    else:
        image, force, proprio, depth, action = [], [], [], [], []
        for i in range(n_time_steps):
            image.append(sample_batched[i][str + "image"])
            force.append(sample_batched[i][str + "force"])
            proprio.append(sample_batched[i][str + "proprio"])
            depth.append(sample_batched[i][str + "depth"])
            action.append(sample_batched[i]["action"])

        # Reshapes to (bs * n, -1) so can parallelize
        image = torch.cat(image, 0)
        force = torch.cat(force, 0)
        proprio = torch.cat(proprio, 0)
        depth = torch.cat(depth, 0)
        action = torch.cat(action, 0)

    # Send it all to CUDA
    if use_cuda:
        image = image.to(device)
        force = force.to(device)
        proprio = proprio.to(device)
        depth = depth.to(device)
        action = action.to(device)

    # Have to add a None to have inputs compatible
    return (image, force, proprio, depth, action, None)


def get_inputs(sample_batched, use_cuda, device, n_time_steps=1):
    """
    Gets the input in the correct format.
    For labels we only want the last time-step
    """
    image, force, proprio, depth, action, _ = get_basic_inputs(
        sample_batched, use_cuda, device, n_time_steps=n_time_steps
    )
    (
        unpaired_image,
        unpaired_force,
        unpaired_proprio,
        unpaired_depth,
        _,
        _,
    ) = get_basic_inputs(
        sample_batched, use_cuda, device, n_time_steps=n_time_steps, str="unpaired_"
    )

    if n_time_steps == 1:
        movement_mask = sample_batched["movement_mask"]

        contact_label = sample_batched["contact_next"]
        optical_flow_label = sample_batched["flow"]
        optical_flow_mask_label = sample_batched["flow_mask"]
        gt_ee_pos_delta = sample_batched["ee_yaw_next"]
    else:
        movement_mask = []
        contact_label, optical_flow_label, optical_flow_mask_label, gt_ee_pos_delta = (
            [],
            [],
            [],
            [],
        )

        for i in range(n_time_steps):
            movement_mask.append(sample_batched[i]["movement_mask"])

            # Labels
            if i == n_time_steps - 1:
                contact_label.append(sample_batched[i]["contact_next"])
                optical_flow_label.append(sample_batched[i]["flow"])
                optical_flow_mask_label.append(sample_batched[i]["flow_mask"])
                gt_ee_pos_delta.append(sample_batched[i]["ee_yaw_next"])

        # Reshapes to (bs * n, -1) so can parallelize

        movement_mask = torch.cat(movement_mask, 0)
        contact_label = torch.cat(contact_label, 0)
        optical_flow_label = torch.cat(optical_flow_label, 0)
        optical_flow_mask_label = torch.cat(optical_flow_mask_label, 0)
        gt_ee_pos_delta = torch.cat(gt_ee_pos_delta, 0)

    # Send it all to CUDA
    if use_cuda:
        movement_mask = movement_mask.to(device)
        contact_label = contact_label.to(device)
        optical_flow_label = optical_flow_label.to(device)
        optical_flow_mask_label = optical_flow_mask_label.to(device)
        gt_ee_pos_delta = gt_ee_pos_delta.to(device)

    # Must make sure that inputs and unpaired inputs have the same number of inputs
    return (
        (image, force, proprio, depth, action, movement_mask),
        (
            unpaired_image,
            unpaired_force,
            unpaired_proprio,
            unpaired_depth,
            action,
            None,
        ),
        (contact_label, optical_flow_label, optical_flow_mask_label, gt_ee_pos_delta),
    )


def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def sample_gaussian(m, v, device):

    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)

    return z


def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8

    return m, v


def product_of_experts(m_vect, v_vect):

    T_vect = 1.0 / v_vect

    mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
    var = 1 / T_vect.sum(2)

    return mu, var


def duplicate(x, rep):

    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def depth_deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, 16, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True
        ),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True),
        nn.LeakyReLU(0.1, inplace=True),
        nn.ConvTranspose2d(
            16, out_planes, kernel_size=4, stride=2, padding=1, bias=True
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def rescaleImage(image, output_size=128, scale=1 / 255.0):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    rescaled_image = image * scale
    return rescaled_image


def filter_depth_old(depth_image):
    depth_image = np.where(depth_image > 1e-7, depth_image, np.zeros_like(depth_image))
    return np.where(depth_image < 2, depth_image, np.zeros_like(depth_image))


def filter_depth(depth_image):
    depth_image = depth_image / 255.0
    depth_image = np.where(depth_image > 1e-7, depth_image, np.zeros_like(depth_image))
    return np.where(depth_image < 2, depth_image, np.zeros_like(depth_image))
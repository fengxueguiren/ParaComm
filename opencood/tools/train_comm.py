# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
import statistics

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description='Train communication-efficient model')
    parser.add_argument('--hypes_yaml', '-y', type=str, required=False,
                        help='yaml file path')
    parser.add_argument('--model_dir', default='',
                        help='resume training path when train_mode=resume')
    parser.add_argument('--fusion_method', '-f', default='intermediate',
                        help='passed to inference after training')
    return parser.parse_args()


def _select_checkpoint_from_dir(model_dir):
    bestval = glob.glob(os.path.join(model_dir, 'net_epoch_bestval_at*.pth'))
    if bestval:
        if len(bestval) > 1:
            bestval = sorted(bestval)
        return bestval[-1], 'bestval'

    epoch_files = glob.glob(os.path.join(model_dir, 'net_epoch*.pth'))
    if epoch_files:
        def extract_epoch(path):
            match = re.findall(r'.*net_epoch(.*).pth.*', path)
            return int(match[0]) if match else 0

        epoch_files = sorted(epoch_files, key=extract_epoch)
        return epoch_files[-1], 'latest'

    raise FileNotFoundError('No checkpoint found in %s' % model_dir)


def _load_pretrained_weights(model, pretrained_path, strict_load=False):
    if os.path.isdir(pretrained_path):
        ckpt_path, ckpt_type = _select_checkpoint_from_dir(pretrained_path)
        print('Using %s checkpoint: %s' % (ckpt_type, ckpt_path))
    else:
        ckpt_path = pretrained_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('Checkpoint not found: %s' % ckpt_path)

    state_dict = torch.load(ckpt_path, map_location='cpu')
    incompatible = model.load_state_dict(state_dict, strict=strict_load)

    if hasattr(incompatible, 'missing_keys'):
        print('missing_keys: %d, unexpected_keys: %d' % (
            len(incompatible.missing_keys), len(incompatible.unexpected_keys)
        ))


def _apply_finetune_policy(model, finetune_cfg):
    default_prefixes = ['dict_comm', 'cls_head', 'reg_head', 'dir_head']
    trainable_prefixes = finetune_cfg.get('trainable_module_prefixes', default_prefixes)
    freeze_unlisted = bool(finetune_cfg.get('freeze_unlisted', True))

    if hasattr(model, 'set_trainable_by_prefix'):
        model.set_trainable_by_prefix(trainable_prefixes, freeze_unlisted=freeze_unlisted)
    else:
        prefix_tuple = tuple(trainable_prefixes)
        for name, param in model.named_parameters():
            matched = name.startswith(prefix_tuple)
            if freeze_unlisted:
                param.requires_grad = matched
            elif matched:
                param.requires_grad = True

    matched_prefixes = set()
    for name, _ in model.named_parameters():
        for prefix in trainable_prefixes:
            if name.startswith(prefix):
                matched_prefixes.add(prefix)

    not_matched = [p for p in trainable_prefixes if p not in matched_prefixes]
    print('Trainable prefixes: %s' % trainable_prefixes)
    if not_matched:
        print('Unmatched prefixes: %s' % not_matched)


def _count_params(model):
    trainable = 0
    frozen = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return trainable, frozen


def _build_optimizer_with_trainable_params(hypes, model):
    method_dict = hypes['optimizer']
    optimizer_method = getattr(torch.optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('%s is not supported' % method_dict['core_method'])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError('No trainable parameters found. Check finetune prefixes.')

    if 'args' in method_dict:
        return optimizer_method(trainable_params,
                                lr=method_dict['lr'],
                                **method_dict['args'])
    return optimizer_method(trainable_params, lr=method_dict['lr'])


def _print_comm_config(comm_cfg,
                       recon_loss_weight,
                       sparse_loss_weight,
                       coherence_loss_weight,
                       budget_loss_weight):
    print('='.ljust(72, '='))
    print('Communication Config')
    print('  enabled=%s, sparse_method=%s, dict_size=%s, atom_topk=%s' % (
        comm_cfg.get('enabled', True),
        comm_cfg.get('sparse_method', 'topk'),
        comm_cfg.get('dict_size', 'N/A'),
        comm_cfg.get('atom_topk', 'N/A'),
    ))
    print('  topk_ratio=%s, topk_min=%s, target_comm_rate=%s' % (
        comm_cfg.get('topk_ratio', 'N/A'),
        comm_cfg.get('topk_min', 'N/A'),
        comm_cfg.get('target_comm_rate', 'N/A'),
    ))
    print('  quant_enabled=%s, coeff_bits=%s, grouped_indexing=%s, group_tile_size=%s, group_atom_count=%s' % (
        comm_cfg.get('enable_coeff_quant', True),
        comm_cfg.get('sparse_coeff_bits', 'N/A'),
        comm_cfg.get('grouped_indexing_enabled', False),
        comm_cfg.get('group_tile_size', 'N/A'),
        comm_cfg.get('group_atom_count', 'N/A'),
    ))
    print('  loss_weights: recon=%.6f sparse=%.6f coherence=%.6f budget=%.6f' % (
        recon_loss_weight,
        sparse_loss_weight,
        coherence_loss_weight,
        budget_loss_weight,
    ))
    print('='.ljust(72, '='))


def _save_val_loss_plot(val_epochs, val_losses, save_path):
    if plt is None:
        return
    if len(val_losses) == 0:
        return

    fig = plt.figure(figsize=(9, 5), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(val_epochs, val_losses, color='#1f77b4', linewidth=2.2, marker='o', markersize=5)
    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    ax.scatter([val_epochs[best_idx]], [val_losses[best_idx]], color='#d62728', s=48, zorder=3)
    ax.annotate('best: %.4f' % val_losses[best_idx],
                (val_epochs[best_idx], val_losses[best_idx]),
                textcoords='offset points',
                xytext=(8, -12),
                fontsize=9)
    ax.set_title('Validation Loss Curve', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'validation_loss_curve.png'))
    plt.close(fig)


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    train_mode = hypes.get('train_mode', 'scratch_comm')
    if train_mode not in ['scratch_comm', 'resume', 'finetune_from_noncomm']:
        raise ValueError('Unsupported train_mode: %s' % train_mode)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=16,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)

    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=16,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    if train_mode == 'resume':
        if not opt.model_dir:
            raise ValueError('train_mode=resume requires --model_dir')
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler_init_epoch = init_epoch
    else:
        init_epoch = 0
        scheduler_init_epoch = None
        saved_path = train_utils.setup_train(hypes)

        if train_mode == 'finetune_from_noncomm':
            finetune_cfg = hypes.get('finetune', {})
            pretrained_noncomm_dir = hypes.get('pretrained_noncomm_dir', '')
            if not pretrained_noncomm_dir:
                raise ValueError('pretrained_noncomm_dir is required for finetune_from_noncomm')

            strict_load = bool(finetune_cfg.get('strict_load', False))
            _load_pretrained_weights(model,
                                     pretrained_noncomm_dir,
                                     strict_load=strict_load)
            _apply_finetune_policy(model, finetune_cfg)

    if torch.cuda.is_available():
        model.to(device)

    trainable_count, frozen_count = _count_params(model)
    print('Trainable params: %d | Frozen params: %d' % (trainable_count, frozen_count))

    criterion = train_utils.create_loss(hypes)
    optimizer = _build_optimizer_with_trainable_params(hypes, model)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=scheduler_init_epoch)
    comm_cfg = hypes.get('model', {}).get('args', {}).get('dict_comm', {})
    recon_loss_weight = float(comm_cfg.get('recon_loss_weight', 0.0))
    sparse_loss_weight = float(comm_cfg.get('sparse_loss_weight', 0.0))
    coherence_loss_weight = float(comm_cfg.get('coherence_loss_weight', 0.0))
    budget_loss_weight = float(comm_cfg.get('budget_loss_weight', 0.0))
    _print_comm_config(comm_cfg,
                       recon_loss_weight,
                       sparse_loss_weight,
                       coherence_loss_weight,
                       budget_loss_weight)

    writer = SummaryWriter(saved_path)
    val_epochs = []
    val_losses = []

    print('Training start with mode: %s' % train_mode)
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, 'supervise_single') else opencood_train_dataset.supervise_single

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group['lr'])

        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum() == 0:
                continue

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            output_dict = model(batch_data['ego'])

            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            recon_raw = output_dict['comm_recon_loss'].item() if 'comm_recon_loss' in output_dict else 0.0
            sparse_raw = output_dict['comm_sparse_loss'].item() if 'comm_sparse_loss' in output_dict else 0.0
            coherence_raw = output_dict['comm_coherence_loss'].item() if 'comm_coherence_loss' in output_dict else 0.0
            budget_raw = output_dict['comm_budget_loss'].item() if 'comm_budget_loss' in output_dict else 0.0
            if ('comm_recon_loss' in output_dict) and (recon_loss_weight > 0.0):
                final_loss = final_loss + recon_loss_weight * output_dict['comm_recon_loss']
            if ('comm_sparse_loss' in output_dict) and (sparse_loss_weight > 0.0):
                final_loss = final_loss + sparse_loss_weight * output_dict['comm_sparse_loss']
            if ('comm_coherence_loss' in output_dict) and (coherence_loss_weight > 0.0):
                final_loss = final_loss + coherence_loss_weight * output_dict['comm_coherence_loss']
            if ('comm_budget_loss' in output_dict) and (budget_loss_weight > 0.0):
                final_loss = final_loss + budget_loss_weight * output_dict['comm_budget_loss']
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(output_dict,
                                        batch_data['ego']['label_dict_single'],
                                        suffix='_single')
                criterion.logging(epoch, i, len(train_loader), writer, suffix='_single')

            if 'comm_recon_loss' in output_dict:
                writer.add_scalar('Comm/recon_loss_raw', output_dict['comm_recon_loss'].item(), epoch * len(train_loader) + i)
                writer.add_scalar('Comm/recon_loss_weighted', (recon_loss_weight * output_dict['comm_recon_loss']).item(), epoch * len(train_loader) + i)
            if 'comm_sparse_loss' in output_dict:
                writer.add_scalar('Comm/sparse_loss_raw', output_dict['comm_sparse_loss'].item(), epoch * len(train_loader) + i)
                writer.add_scalar('Comm/sparse_loss_weighted', (sparse_loss_weight * output_dict['comm_sparse_loss']).item(), epoch * len(train_loader) + i)
            if 'comm_coherence_loss' in output_dict:
                writer.add_scalar('Comm/coherence_loss_raw', output_dict['comm_coherence_loss'].item(), epoch * len(train_loader) + i)
                writer.add_scalar('Comm/coherence_loss_weighted', (coherence_loss_weight * output_dict['comm_coherence_loss']).item(), epoch * len(train_loader) + i)
            if 'comm_budget_loss' in output_dict:
                writer.add_scalar('Comm/budget_loss_raw', output_dict['comm_budget_loss'].item(), epoch * len(train_loader) + i)
                writer.add_scalar('Comm/budget_loss_weighted', (budget_loss_weight * output_dict['comm_budget_loss']).item(), epoch * len(train_loader) + i)

            print('[Train][Epoch %d/%d][Iter %d/%d] total=%.6f | recon=%.6f(%.6f) sparse=%.6f(%.6f) coherence=%.6f(%.6f) budget=%.6f(%.6f)' % (
                epoch + 1,
                epoches,
                i + 1,
                len(train_loader),
                final_loss.item(),
                recon_raw,
                recon_loss_weight * recon_raw,
                sparse_raw,
                sparse_loss_weight * sparse_raw,
                coherence_raw,
                coherence_loss_weight * coherence_raw,
                budget_raw,
                budget_loss_weight * budget_raw,
            ))

            final_loss.backward()
            optimizer.step()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            with torch.no_grad():
                for _, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue

                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    output_dict = model(batch_data['ego'])

                    final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    if ('comm_recon_loss' in output_dict) and (recon_loss_weight > 0.0):
                        final_loss = final_loss + recon_loss_weight * output_dict['comm_recon_loss']
                    if ('comm_sparse_loss' in output_dict) and (sparse_loss_weight > 0.0):
                        final_loss = final_loss + sparse_loss_weight * output_dict['comm_sparse_loss']
                    if ('comm_coherence_loss' in output_dict) and (coherence_loss_weight > 0.0):
                        final_loss = final_loss + coherence_loss_weight * output_dict['comm_coherence_loss']
                    if ('comm_budget_loss' in output_dict) and (budget_loss_weight > 0.0):
                        final_loss = final_loss + budget_loss_weight * output_dict['comm_budget_loss']
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            val_epochs.append(epoch + 1)
            val_losses.append(valid_ave_loss)
            _save_val_loss_plot(val_epochs, val_losses, saved_path)
            print('+' + '-' * 70 + '+')
            print('| Validation Summary | epoch=%3d | loss=%10.6f | best=%10.6f |' % (
                epoch + 1,
                valid_ave_loss,
                min(val_losses),
            ))
            print('+' + '-' * 70 + '+')
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                           os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                last_best = os.path.join(saved_path,
                                         'net_epoch_bestval_at%d.pth' % lowest_val_epoch)
                if lowest_val_epoch != -1 and os.path.exists(last_best):
                    os.remove(last_best)
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        scheduler.step(epoch)
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True
    bestval_model_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*'))
    if len(bestval_model_list) > 1:
        import numpy as np
        bestval_model_epoch_list = [
            eval(x.split('/')[-1].lstrip('net_epoch_bestval_at').rstrip('.pth'))
            for x in bestval_model_list
        ]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    if run_test:
        # clear GPU cache before inference to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fusion_method = opt.fusion_method
        if 'noise_setting' in hypes and hypes['noise_setting']['add_noise']:
            cmd = 'python opencood/tools/inference_w_noise.py --model_dir %s --fusion_method %s' % (
                saved_path, fusion_method
            )
        else:
            cmd = 'python opencood/tools/inference.py --model_dir %s --fusion_method %s' % (
                saved_path, fusion_method
            )
        print('Running command: %s' % cmd)
        os.system(cmd)


if __name__ == '__main__':
    main()

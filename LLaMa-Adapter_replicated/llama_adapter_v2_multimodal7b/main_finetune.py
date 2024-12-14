import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_adapter import LLaMA_adapter

from data.dataset import FinetuneDataset, transform_train

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

from engine_finetune import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('llama_adapterV2 pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=512, type=int,
                        help='max number of input words')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='configs/data/finetune/EN.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--enable_distributed', action='store_true', 
                        help='Enable distributed training')

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--distributed', action='store_true', 
                        help='Enable distributed training')
  
    return parser


def main(args):
    # Initialize distributed mode if required
    if args.distributed:
        misc.init_distributed_mode(args)
    else:
        print("Running in non-distributed mode.")

    # Print job directory and arguments
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:")
    print("{}".format(args).replace(', ', ',\n'))

    # Set the device
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Set random seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path)
    model.to(device)

    # Print model details
    print("Model:")
    print(model)
    print("Trainable Parameters:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    # Handle distributed training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module

    # Calculate effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # Compute learning rate if not explicitly set
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base learning rate: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual learning rate: {args.lr:.2e}")
    print(f"Gradient accumulation iterations: {args.accum_iter}")
    print(f"Effective batch size: {eff_batch_size}")

    # Optimizer and loss scaler
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("Optimizer:")
    print(optimizer)
    loss_scaler = NativeScaler()

    # Load pretrained model checkpoint
    misc.load_model(model_without_ddp, args.pretrained_path)

    # Prepare dataset and sampler
    dataset_train = FinetuneDataset(args.data_config, transform=transform_train,
                                    max_words=args.max_words, tokenizer_path=llama_tokenzier_path)
    print(f"Loaded training dataset: {dataset_train}")

    if args.enable_distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Set up TensorBoard writer
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Start training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Save the model after every epoch
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

        # Log training statistics
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            if log_writer is not None:
                log_writer.flush()

    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

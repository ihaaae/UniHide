import argparse
from os.path import join

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from loguru import logger

from trainer_single import Trainer
from dataset_single import TimitDataset
from hparams import *
from model_single import FullEncoder, MsgDecoder

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_dataloader(train_path, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    train_dataset = TimitDataset(train_path,
                                n_pairs     = 4608,
                                # n_pairs=32,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size  = batch_size,
                                  shuffle     = True,
                                  num_workers = num_workers)
    
    return train_dataloader

def val_dataloader(val_path, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    val_dataset = TimitDataset(val_path,
                               n_pairs     = 832,
                            #    n_pairs=32,
                               trim_start  = trim_start,
                               num_samples = num_samples)
    val_dataloader = DataLoader(val_dataset,
                                batch_size  = batch_size,
                                shuffle     = False,
                                num_workers = num_workers)
    
    return val_dataloader

def test_dataloader(test_path, batch_size):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    test_dataset = TimitDataset(test_path,
                                n_pairs     = 832,
                                # n_pairs=32,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size  = batch_size,
                                 shuffle     = False,
                                 num_workers = 0)
    
    return test_dataloader

def load_models(encoder, decoder, ckpt_dir):
    encoder.load_state_dict(torch.load(join(ckpt_dir, "encoder.ckpt")))
    decoder.load_state_dict(torch.load(join(ckpt_dir, "decoder.ckpt")))
    logger.info("loader models")

def get_models(hparams):
        dec_m_conv_dim  = 1
        dec_um_conv_dim = 1 + 64

        encoder = FullEncoder(hparams.block_type, hparams.enc_n_layers, dec_um_conv_dim, hparams.dec_c_n_layers).to(device)

        decoder =  MsgDecoder(conv_dim=dec_m_conv_dim, block_type=hparams.block_type).to(device)
        
        params = list(encoder.parameters()) + list(decoder.parameters())
        
        optimizer = torch.optim.Adam(params, lr=hparams.lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        if hparams.load_ckpt:
            load_models(encoder, decoder, hparams.load_ckpt)

        logger.debug(encoder)
        logger.debug(decoder)

        return encoder, decoder, optimizer, scheduler

def run(hparams):
    torch.set_num_threads(1000)
    trainer = Trainer(hparams)
    
    encoder, decoder, optimizer, scheduler = get_models(hparams)

    if hparams.mode == 'train':
        train_loader = train_dataloader(hparams.train_path, hparams.batch_size, hparams.num_workers)
        val_loader   = val_dataloader(hparams.val_path, hparams.batch_size, hparams.num_workers)

        logger.info(f"loaded train ({len(train_loader)}), val ({len(val_loader)})")

        trainer.train(train_loader, val_loader, encoder, decoder, optimizer, scheduler)
    elif hparams.mode == 'test':
        test_loader = test_dataloader(hparams.test_path, hparams.batch_size)

        logger.info(f"loaded test ({len(test_loader)})")

        trainer.test(test_loader, encoder, decoder)
    elif hparams.mode == 'sample':
        pass
        # solver.eval_mode()
        # solver.sample_examples()

def main():
    parser = argparse.ArgumentParser(description='Hide and Speak')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'abs'], help='loss function used for training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'], help='`train` will initiate training, `test` should be used in conjunction with `load_ckpt` to run a test epoch, `sample` should be used in conjunction with `load_ckpt` to sample examples from dataset')
    parser.add_argument('--train_path', required=True, type=str, help='path to training set. should be a folder containing .wav files for training')
    parser.add_argument('--val_path', required=True, type=str, help='')
    parser.add_argument('--test_path', required=True, type=str, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_pairs', type=int, default=32, help='number of training examples generated from wav files')
    parser.add_argument('--dataset', type=str, default='timit', help='select dataset', choices=['timit', 'mini'])
    parser.add_argument('--message_file', type=str, default='', help='the absolute path of the secrete message file')

    parser.add_argument('--block_type', type=str, default='normal', choices=['normal', 'skip', 'bn', 'in', 'relu'], help='type of block for encoder/decoder')
    parser.add_argument('--enc_n_layers', default=3, type=int, help='number of layers in encoder')
    parser.add_argument('--dec_c_n_layers', default=4, type=int, help='number of layers in decoder')
    parser.add_argument('--lambda_carrier_loss', type=float, default=3.0, help='coefficient for carrier loss term')
    parser.add_argument('--lambda_msg_loss', type=float, default=1.0, help='coefficient for message loss term')

    parser.add_argument('--num_workers', type=int, default=20, help='number of data loading workers')
    parser.add_argument('--load_ckpt', type=str, default=None, help='path to checkpoint (used for test epoch or for sampling)')
    parser.add_argument('--run_dir', type=str, default='.', help='output directory for logs, samples and checkpoints')
    hparams = parser.parse_args()
    run(hparams)

if __name__ == '__main__':
    main()
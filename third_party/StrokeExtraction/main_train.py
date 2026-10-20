import argparse

import torch

from train_SDNet import TrainSDNet
from train_SegNet import TrainSegNet
from train_ExtractNet import TrainExtractNet


class MainTrain():
    def __init__(self, dataset='RHSEDB', sdnet_epochs=80, segnet_epochs=20,
                 extractnet_epochs=40, sdnet_batch_size=16,
                 segnet_batch_size=32, extractnet_batch_size=32,
                 num_workers=4, sdnet_lr=1e-5, segnet_lr=1e-5,
                 extractnet_lr=1e-5, init_sdnet=None, init_segnet=None,
                 init_extractnet=None, plot_losses=True):
        '''
        Select the dataset to train from ['CCSEDB', 'RHSEDB']
        '''
        self.dataset = dataset
        self.sdnet_epochs = sdnet_epochs
        self.segnet_epochs = segnet_epochs
        self.extractnet_epochs = extractnet_epochs
        self.sdnet_batch_size = sdnet_batch_size
        self.segnet_batch_size = segnet_batch_size
        self.extractnet_batch_size = extractnet_batch_size
        self.num_workers = num_workers
        self.sdnet_lr = sdnet_lr
        self.segnet_lr = segnet_lr
        self.extractnet_lr = extractnet_lr
        self.plot_losses = plot_losses
        self.train_sdnet = TrainSDNet(save_path='out/SDNet', dataset=dataset,
                                      init_model=init_sdnet)
        self.train_segnet = TrainSegNet(save_path='out/SegNet_'+dataset,
                                        init_model=init_segnet)
        self.train_extractnet = TrainExtractNet(
            save_path='out/ExtractNet_'+dataset,
            segNet_save_path='out/SegNet_'+dataset,
            init_model=init_extractnet)

    def train(self):
        # train SDNet
        self.train_sdnet.train_model(
            epochs=self.sdnet_epochs, init_learning_rate=self.sdnet_lr,
            batch_size=self.sdnet_batch_size, lr_step_size=20,
            num_workers=self.num_workers, plot_losses=self.plot_losses)
        print('SDNet training has been completed')
        # get prior information and other data for SegNet and ExtractNet
        self.train_sdnet.calculate_prior_information_and_qualitative(save_path='dataset_forSegNet_ExtractNet_'+self.dataset)
        print('calculating prior information has been completed')
        # train SegNet
        self.train_segnet.train_model(
            epochs=self.segnet_epochs, init_learning_rate=self.segnet_lr,
            batch_size=self.segnet_batch_size, lr_step_size=4,
            num_workers=self.num_workers,
            dataset_path='dataset_forSegNet_ExtractNet_'+self.dataset,
            plot_losses=self.plot_losses)
        print('SegNet training has been completed')
        #train ExtractNet
        self.train_extractnet.train_model(
            epochs=self.extractnet_epochs, init_learning_rate=self.extractnet_lr,
            batch_size=self.extractnet_batch_size, lr_step_size=10,
            num_workers=self.num_workers,
            dataset='dataset_forSegNet_ExtractNet_'+self.dataset,
            plot_losses=self.plot_losses)
        print('ExtractNet training has been completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train all three stages; defaults target an RTX 5090 (32 GB).')
    parser.add_argument('--dataset', default='RHSEDB',
                        choices=['CCSEDB', 'RHSEDB', 'MMAH_CCSEDB_v4'])
    parser.add_argument('--sdnet-epochs', type=int, default=80)
    parser.add_argument('--segnet-epochs', type=int, default=20)
    parser.add_argument('--extractnet-epochs', type=int, default=40)
    parser.add_argument('--sdnet-batch-size', type=int, default=16)
    parser.add_argument('--segnet-batch-size', type=int, default=32)
    parser.add_argument('--extractnet-batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4)')
    parser.add_argument('--sdnet-lr', type=float, default=1e-5)
    parser.add_argument('--segnet-lr', type=float, default=1e-5)
    parser.add_argument('--extractnet-lr', type=float, default=1e-5)
    parser.add_argument('--init-sdnet')
    parser.add_argument('--init-segnet')
    parser.add_argument('--init-extractnet')
    parser.add_argument('--no-loss-plots', action='store_true',
                        help='Save numeric loss history.npz without calling matplotlib during training')
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print('Training on logical cuda:{} ({})'.format(
            torch.cuda.current_device(), torch.cuda.get_device_name()))
    print('Training parameters: SDNet epochs/batch={}/{}, '
          'SegNet={}/{}, ExtractNet={}/{}'.format(
              args.sdnet_epochs, args.sdnet_batch_size,
              args.segnet_epochs, args.segnet_batch_size,
              args.extractnet_epochs, args.extractnet_batch_size))
    train_ = MainTrain(
        dataset=args.dataset,
        sdnet_epochs=args.sdnet_epochs,
        segnet_epochs=args.segnet_epochs,
        extractnet_epochs=args.extractnet_epochs,
        sdnet_batch_size=args.sdnet_batch_size,
        segnet_batch_size=args.segnet_batch_size,
        extractnet_batch_size=args.extractnet_batch_size,
        num_workers=args.num_workers,
        sdnet_lr=args.sdnet_lr,
        segnet_lr=args.segnet_lr,
        extractnet_lr=args.extractnet_lr,
        init_sdnet=args.init_sdnet,
        init_segnet=args.init_segnet,
        init_extractnet=args.init_extractnet,
        plot_losses=not args.no_loss_plots)
    train_.train()

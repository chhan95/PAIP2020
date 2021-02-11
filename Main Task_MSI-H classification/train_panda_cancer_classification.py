import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer,EarlyStopping
from ignite.metrics import RunningAverage
from imgaug import augmenters as iaa
from tensorboardX import SummaryWriter

import dataset
from config import Config
#from .misc.train_utils import *
from train_utils import *
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import StepLR
from ignite.contrib.handlers import LRScheduler
import os
class Trainer(Config):
    ####
    def view_dataset(self, mode='train'):
        train_pairs, valid_pairs = dataset.prepare_PAIP2020_PANDA(self.fold_idx)
        if mode == 'train':
            train_augmentors = self.train_augmentors()
            ds = dataset.DatasetSerial(train_pairs,self.tile_size,self.num_tile,True)
        else:
            infer_augmentors = self.infer_augmentors()  # HACK
            ds = dataset.DatasetSerial(valid_pairs,self.tile_size,self.num_tile,False)
        dataset.visualize(ds, 1)
        return

    ####
    def train_step(self, net, batch, optimizer, device):
        net.train()  # train mode

        imgs_cpu, true_cpu = batch
        imgs_cpu = imgs_cpu.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs
        imgs = imgs_cpu.to(device).float()
        true = true_cpu.to(device).long()  # not one-hot

        # -----------------------------------------------------------
        net.zero_grad()  # not rnn so not accumulate

        logit ,feature_vector= net(imgs)  # forward

        #  logit, aux_logit = net(imgs) # forward
        prob = F.softmax(logit, dim=-1)

        # has built-int log softmax so accept logit
        weight = torch.from_numpy(np.array([0.5, 1.0])).to('cuda').float()
        loss = F.cross_entropy(prob, true, reduction='mean',weight=weight)


        pred = torch.argmax(prob, dim=-1)
        acc = torch.mean((pred == true).float())  # batch accuracy

        # gradient update
        loss.backward()
        optimizer.step()

        # -----------------------------------------------------------
        return dict(
            loss=loss.item(),
            acc=acc.item(),
        )

    ####
    def infer_step(self, net, batch, device):
        net.eval()  # infer mode

        imgs, true = batch
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long()  # not one-hot

        # -----------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            logit,feature_vector = net(imgs)
            prob = nn.functional.softmax(logit, dim=-1)
            return dict(prob=prob.cpu().numpy(),
                        true=true.cpu().numpy())

    ####
    def run_once(self):
        # self.log_path = 'log/%s/' % self.dataset
        # self.model_name = 'efficientnet-b0_MSI_{0}fold_random_tile_patch'.format(self.fold_idx)
        # self.log_dir = self.log_path + self.model_name

        log_dir = self.log_dir
        check_manual_seed(self.seed)
        train_pairs, valid_pairs =dataset.prepare_PAIP2020_PANDA(self.fold_idx)
        print(len(train_pairs))
        print(len(valid_pairs))

        train_augmentors = self.train_augmentors()
        train_dataset = dataset.DatasetSerial(train_pairs[:],self.tile_size,self.num_tile,train_mode=True)

        infer_augmentors = self.infer_augmentors()  # HACK at has_aux
        infer_dataset = dataset.DatasetSerial(valid_pairs[:],self.tile_size,self.num_tile,train_mode=False)

        train_loader = data.DataLoader(train_dataset,
                                       num_workers=self.nr_procs_train,
                                       batch_size=self.train_batch_size,
                                       shuffle=True, drop_last=True)

        valid_loader = data.DataLoader(infer_dataset,
                                       num_workers=self.nr_procs_valid,
                                       batch_size=self.infer_batch_size,
                                       shuffle=True, drop_last=False)

        # --------------------------- Training Sequence

        if self.logging:
            check_log_dir(log_dir)
        #
        device = 'cuda'

        # networksv
        input_chs = 3  # TODO: dynamic config
        # ### VGGNet

        net= EfficientNet.from_pretrained('efficientnet-b0',num_classes=2)

        #net =DenseNet(3,2)
        # load pre-trained models
        net = torch.nn.DataParallel(net).to(device)

        if self.load_network:
            saved_state = torch.load(self.save_net_path)
            net.load_state_dict(saved_state)


        # optimizers
        optimizer = optim.Adam(net.parameters(), lr=self.init_lr)
        scheduler = StepLR(optimizer, self.lr_steps,gamma=0.1)
        scheduler= LRScheduler(scheduler)
        #
        trainer = Engine(lambda engine, batch: self.train_step(net, batch, optimizer, device))
        valider = Engine(lambda engine, batch: self.infer_step(net, batch, device))

        infer_output = ['prob', 'true']
        ##

        if self.logging:
            checkpoint_handler = ModelCheckpoint(log_dir, self.chkpts_prefix,
                                                 save_interval=1, n_saved=100, require_empty=False)
            # adding handlers using `trainer.add_event_handler` method API
            trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                                      to_save={'net': net})

        timer = Timer(average=True)
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(valider, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        # TODO: refactor this
        RunningAverage(alpha=0.95, output_transform=lambda x: x['acc']).attach(trainer, 'acc')
        RunningAverage(alpha=0.95, output_transform=lambda x: x['loss']).attach(trainer, 'loss')

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=['loss'])
        pbar.attach(valider)

        # #early Stopping
        # def score_function(engine):
        #     val_acc=engine.state.metrics["valid-acc"]
        #     return val_acc
        # early_stopping_handler=EarlyStopping(patience=10,score_function=score_function,trainer=trainer)


        # adding handlers using `trainer.on` decorator API
        @trainer.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
                engine.terminate()
                warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
                checkpoint_handler(engine, {'net_exception': net})
            else:
                raise e

        # writer for tensorboard logging
        tfwriter = None  # HACK temporary
        if self.logging:
            tfwriter = SummaryWriter(log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file)  # create empty file

        ### TODO refactor again
        log_info_dict = {
            'logging': self.logging,
            'optimizer': optimizer,
            'tfwriter': tfwriter,
            'json_file': json_log_file,
            'nr_classes': self.nr_classes,
            'metric_names': infer_output,
            'infer_batch_size': self.infer_batch_size  # too cumbersome
        }


        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_ema_results, log_info_dict)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, inference, valider, valid_loader, log_info_dict)
        valider.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)

        # Setup is done. Now let's run the training
        trainer.run(train_loader, self.nr_epochs)
        return

    ####
    def run(self,):
        self.run_once()
        return


####
if __name__ == '__main__':
    trainer = Trainer()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    trainer.view_dataset(mode="train")
    trainer.run()

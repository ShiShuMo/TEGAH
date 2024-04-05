import warnings
import torch


class Default(object):

    # visualization
    vis_env = 'main'
    vis_port = 8097  # visdom port
    flag = 'mir'

    batch_size = 80
    image_dim = 4096
    hidden_dim = 2048
    modals = 2
    valid = True  # whether to use validation
    valid_freq = 1
    max_epoch = 300
    optim = 'AdamW'
    scheduler = 'my'

    bit = 64  # hash code length
    # lr = 1e-04  # initial learning rate
    # img_lr = 1e-4
    # txt_lr = 1e-4
    # gat_lr = 1e-4
    img_lr = 1e-5
    txt_lr = 1e-4
    gat_lr = 1e-5
    label_lr = 1e-5
    encoder_lr = 1e-4
    decoder_lr = 1e-4

    lr = 1e-04  # initial learning rate
    # dropout = True
    dropout = False
    # img_weight_decay = 1e-5
    # txt_weight_decay = 5e-4
    # gat_weight_decay = 5e-4
    img_weight_decay = 1e-4
    txt_weight_decay = 1e-5
    gat_weight_decay = 1e-4
    label_weight_decay = 1e-4
    encoder_weight_decay = 1e-4
    decoder_weight_decay = 1e-4

    device = 0

    # hyper-parameters
    alpha = 10
    gamma = 0.01
    beta = 1
    mu = 0.5
    rho = 1.05
    deta = 0.3

    margin = 0.4

    def data(self, flag):
        if flag == 'mir':
            self.dataset = 'flickr25k'
            self.data_path = './data/FLICKR-25K/'
            # self.data_path = './data/FLICKR-25K.mat'
            self.db_size = 18015
            self.num_label = 24
            self.query_size = 2000
            self.text_dim = 1386
            self.training_size = 10000
            self.seed = 6
            self.img_dir = '/dev/shm/mirflickr'
            self.adj = './data/FLICKR-25K/mir_adj.mat'
            self.inp = './data/FLICKR-25K/mir_glove.mat'

        elif flag == 'nus':
            self.dataset = 'nus-wide'
            self.data_path = './data/NUS-WIDE-TC10/'
            # self.data_path = './data/NUS-WIDE-TC10.mat'
            self.db_size = 184477
            self.num_label = 10
            self.query_size = 2100
            self.text_dim = 1000
            self.training_size = 10500
            self.seed = 7
            self.img_dir = '/dev/shm/nuswide/Flickr'
            self.adj = './data/NUS-WIDE-TC10/nus_adj.mat'
            self.inp = './data/NUS-WIDE-TC10/nus_glove.mat'

        else:
            self.dataset = 'coco2014'
            self.data_path = './data/coco2014/'
            self.db_size = 117218
            self.num_label = 80
            self.query_size = 5000
            self.text_dim = 2026
            self.training_size = 10000
            self.seed = 8
            self.img_dir = '/dev/shm/coco2014/img'
            self.adj = './data/coco2014/coco_adj.mat'
            self.inp = './data/coco2014/coco_glove.mat'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if k == 'flag':
                self.data(v)
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)
        if self.rank == 0:
            print('Configuration:')
            for k, v in self.__class__.__dict__.items():
                if not k.startswith('__') and str(k) != 'parse' and str(k) != 'data':
                    print('\t{0}: {1}'.format(k, getattr(self, k)))


opt = Default()
if __name__ == '__main__':
    opt = Default()
    print((10 ** (-6) / opt.lr) ** (1 / opt.max_epoch))

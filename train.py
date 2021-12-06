import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import cv2
import csv
import time
import configparser
import pandas as pd
import math
from torch.utils.data import Dataset
from scipy import ndimage
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
#import torchvision.models as models
from ResNet1D import *
#This code is classification based on dynamic bins
#cross-validation: 10-folder datasets

""" SET UP PARAMETERS """

print("Start training... ")

config = configparser.ConfigParser()
config.read(sys.argv[1])
print("Read in parameter file... %s" % sys.argv[1])

MAIN_PATH = config['main']['path']

TRAIN_MAIN_PATH = os.path.join(MAIN_PATH, config['data']['train_folder'])
LABEL_PATH        = config['data']['label_folder']
SPEC_PATH         = config['data']['spec_folder']
MODEL_OUTPUT_PATH = config['data']['output_folder']

LOG_PATH = os.path.join(config['log']['path'], config['log']['folder_name'])

PRETRAIN_FOLD  = config['pretrain']['fold']
PRETRAIN_MODEL = config['pretrain']['model']
PRETRAIN_PATH = os.path.join(MAIN_PATH,
                             config['pretrain']['pre_folder'],
                             'folder-%s' % PRETRAIN_FOLD,
                             'model_%s.pth' % PRETRAIN_MODEL)

TEST_NUM   = int(config['hypar']['test_num'])
EPOCHS     = int(config['hypar']['epochs'])
BATCH_SIZE = int(config['hypar']['batch'])
NUM_CLASS = int(config['hypar']['num_classes'])

PRETRAIN = config.getboolean('switch', 'PRETRAIN')
print("Configures loaded... ")

""" DONE """

print('Test: %d ;' % TEST_NUM, end='')
TEST_LABEL_PATH = os.path.join(LABEL_PATH, str(TEST_NUM) + '.csv')

if TEST_NUM == 9: VAL_NUM = 0
else: VAL_NUM = TEST_NUM + 1
print(' Validation: %d.' % VAL_NUM)
VAL_LABEL_PATH = os.path.join(LABEL_PATH, str(VAL_NUM) + '.csv')

filenames = []
for i in range(10):
    filenames.append(os.path.join(LABEL_PATH, '%d.csv' % i))

#import pdb; pdb.set_trace()
filenames.remove(VAL_LABEL_PATH)
#filenames.remove(TEST_LABEL_PATH)
print('Remove test and validation set.')

comb_train_fn = 'combined_train.csv'
print('Combine training set and write to %s...' % comb_train_fn, end='')
with open(os.path.join(LABEL_PATH, comb_train_fn), 'w') as combined_train_list:
    for fold in filenames:
        for line in open(fold, 'r'):
            combined_train_list.write(line)
print(' Done')
TRAIN_LABEL_PATH = os.path.join(LABEL_PATH, comb_train_fn)

MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_PATH, 'folder-'+str(TEST_NUM))

if not os.path.exists(MODEL_OUTPUT_PATH):
    os.makedirs(MODEL_OUTPUT_PATH)

def read_csv(data_file_path):
    data = []
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.asarray(data)
    return data

class AstroDataset(Dataset):
    def __init__(self, 
                 path = SPEC_PATH,
                 label_file = TRAIN_LABEL_PATH):
        self.dir = path
        self.files = read_csv(label_file)

    def __getitem__(self,idx):
        temp = self.files[idx]
        spec = np.load(os.path.join(self.dir, '{}.npy'.format(int(float(temp[0])))))
        spec = (spec - spec.min())/np.ptp(spec) + 1e-5
        l1 = np.float32(temp[1:])
        #l2 = np.float32(temp[2])
        return spec, l1#, l2

    def __len__(self):
        return len(self.files)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_set = AstroDataset()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=4,
                                           drop_last = True)

test_set = AstroDataset(label_file = VAL_LABEL_PATH)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=4,
                                          drop_last = True)

model = resnet18(num_classes=NUM_CLASS, pretrained=False)
model = model.to(device)

if PRETRAIN:
    model.load_state_dict(torch.load(PRETRAIN_PATH))

#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
#scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001,
#                                        max_lr=0.1, cycle_momentum=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01, last_epoch=-1)

criterion = nn.MSELoss()

def assemble_labels(step, y_true, y_pred, label, out):
    if(step==0):
        y_true = label
        y_pred = out
    else:
        y_true = torch.cat((y_true, label), 0)
        y_pred = torch.cat((y_pred, out), 0)

    return y_true, y_pred

loss_tag = ['test v loss',
            'test z loss',
            'test T loss',
            'test a loss',
            'test n loss']
epoch_loss_tag = ['test epoch v loss',
                  'test epoch z loss',
                  'test epoch T loss',
                  'test epoch a loss',
                  'test epoch n loss']
abs_loss_tag = ['mean abs v loss',
                'mean abs z loss',
                'mean abs T loss',
                'mean abs a loss',
                'mean abs n loss']
label_name = ['velocity', 'redshift', 'temperature', 'abundance', 'normalization']

writer=SummaryWriter(os.path.join(LOG_PATH, 'fold-%d' % TEST_NUM))
iter_train = 0
iter_test = 0
for epoch in range(EPOCHS): 

    count = 0
    test_loss = 0
    test_epoch_loss = 0

    test_epoch_loss_each = [0] * NUM_CLASS
    abs_total_each = [0] * NUM_CLASS
    abs_loss_each = [0] * NUM_CLASS
    test_true = []
    test_hat = []
    for i in range(NUM_CLASS):
        test_true.append([])
        test_hat.append([])

    test_true_out = []
    test_hat_out = []

    model.train()    
    for step, (x, y) in enumerate(train_loader):              
        b_x = x.clone().float().to(device)
        b_x = torch.unsqueeze(b_x, 1)
        b_y = y.clone().float().to(device)
        output = model(b_x)
        output = torch.squeeze(output, 1)

        loss = criterion(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch = ', epoch, 'training loss:', loss.item())
        writer.add_scalar('training loss', loss.item(), iter_train)
        iter_train += 1
    writer.add_scalar('training epoch loss', loss.item()/(step+1), epoch)

    scheduler.step()

    model.eval()
    count = 0
    acc = 0
    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(test_loader):
            test_b_x = torch.tensor(test_x).float().to(device)
            test_b_x = torch.unsqueeze(test_b_x, 1)
            test_b_y = torch.tensor(test_y).float().to(device)

            test_output = model(test_b_x)
            test_output = torch.squeeze(test_output)

            test_loss = criterion(test_output, test_b_y)
            writer.add_scalar('test loss', test_loss.item(), iter_test)
            for i in range(NUM_CLASS):
                tmp = criterion(test_output[:,i], test_b_y[:,i])
                test_epoch_loss_each[i] += tmp
                writer.add_scalar(loss_tag[i], tmp.item(), iter_test)

            test_epoch_loss += test_loss

            abs_batch_sum = torch.abs(test_output - test_b_y)
            for i in range(NUM_CLASS):
                tmp = abs_batch_sum.mean(axis=0)[i]
                abs_total_each[i] += tmp.item()

            for i in range(NUM_CLASS):
                test_true[i], test_hat[i] = assemble_labels(step,
                                                            test_true[i],
                                                            test_hat[i],
                                                            test_b_y[:,i],
                                                            test_output[:,i])

            count += 1
            iter_test += 1

    writer.add_scalar('test epoch loss', test_epoch_loss.item()/(step+1), epoch)
    for i in range(NUM_CLASS):
        print('Epoch = ', epoch,
              'test loss ({}):'.format(label_name[i]),
              test_epoch_loss_each[i].item())
        writer.add_scalar(epoch_loss_tag[i], test_epoch_loss_each[i].item()/(step+1), epoch)

        abs_loss_each[i] = abs_total_each[i] / count
        print('mean abs loss ({}):'.format(label_name[i]), abs_loss_each[i])
        writer.add_scalar(abs_loss_tag[i], abs_loss_each[i], epoch)

        test_true_out.append(test_true[i].detach().cpu().numpy())
        test_hat_out.append(test_hat[i].detach().cpu().numpy())
    np.save(os.path.join(MODEL_OUTPUT_PATH, 'test_y_true_'+str(epoch)+'.npy'),
            np.array(test_true_out))
    np.save(os.path.join(MODEL_OUTPUT_PATH, 'test_y_hat_'+str(epoch)+'.npy'),
            np.array(test_hat_out))
    torch.save(model.state_dict(), '%s/model_%d.pth' % (MODEL_OUTPUT_PATH, epoch))

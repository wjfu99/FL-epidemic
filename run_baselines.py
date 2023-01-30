import numpy as np
# from modules.models import MultiScaleFedGNN
from modules import MultiScaleFedGNN
# import modules
from Agent_Epi_Sim import Engine
from utils import hypergraph_generator, hypergraph_sequence_generator, label_generator, hypergraph2hyperindex, construct_network
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import os
import configparser
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve
from utils.fake_loc_generator import fake_loc_gen, plausible_loc_gen
from utils.dp_lib import usr_emb_clip, fl_dp
import copy
from datetime import datetime
import joblib
from baselines.GNN import GCN
import torch_geometric
from scipy.sparse import coo_array

# load the cfg file
# cfg = configparser.ConfigParser()
# cfg.read('cfg.ini')
with open("bs_config.json", 'r') as f:
    cfg = json.load(f)
fun_args, env_args, model_args, optim_args\
    = cfg['fun_args'], cfg['env_args'], cfg['model_args'], cfg['optim_args']

current_time = datetime.now()

# For more specific debugging results.
if cfg['fun_args']["debug"]:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cpu")
else:
    # Define the default GPU device
    device = torch.device("cuda:0")
if fun_args['tensorboard']:
    writer = SummaryWriter(log_dir='./runs/'+cfg['fun_args']['tsboard_comm']+current_time.strftime('%m-%d %H:%M'))

# Process the epidemic data.
if env_args['dataset'] == 'old_data':
    data_path = './datasets/beijing/small-unfilled-unclustered/'
    traj = np.load(data_path + "traj_mat.npy")
    usr_num = traj.shape[0]
    lbls = np.load(data_path + 'label_omicron.npy')
    lbls = torch.tensor(lbls).to(device).squeeze()
    env_args['sim_days'] = 40
elif env_args['dataset'] == 'largec': # chose as the benchmark.
    data_path = './datasets/beijing/large-filled-clustered/'
    traj = np.load(data_path + "traj_mat(filled,sample).npy")
    usr_num = traj.shape[0]
    lbls = np.load(data_path + 'label_omicron.npy')
    lbls = torch.tensor(lbls).to(device).squeeze()
    env_args['sim_days'] = 14

train_ratio = cfg['env_args']['train_ratio']
sample_num = lbls.shape[0]
train_num = int(sample_num*train_ratio)
idx_train = np.array(range(train_num))
idx_test = np.array(range(train_num, sample_num))


edge_index = construct_network(traj[:, :env_args['sim_days']*48])
edge_index = edge_index.to(device)

# edge_index = np.load(data_path + "adj.npy")
# adj = edge_index
# adj = coo_array(adj)
# edge_index, edge_attr = torch_geometric.utils.from_scipy_sparse_matrix(adj)
# edge_index = edge_index.to(device)


# Model definition
model = GCN(usr_num=usr_num).to(device)
summary(model)
criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 3]).to(device))
optimizer = optim.Adam(model.parameters(), lr=cfg['optim_args']["lr"], weight_decay=cfg['optim_args']["weight_decay"]) # TODO: SGD for FL?
schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['optim_args']["milestones"], gamma=cfg['optim_args']['gamma'])




def train_model(model, criterion, optimizer, scheduler, num_epochs, print_freq=10):

    for epoch in range(num_epochs):

        # init parameters
        loss_val = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            idx = idx_train if phase == 'train' else idx_test
            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # outputs = model(fts, support)
                outputs = model(edge_index)
                loss = criterion(outputs[idx], lbls[idx])

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    # clip on the user embedding.
                    if cfg['model_args']['fl_dp']:
                        fl_dp(model, optimizer, **cfg['model_args'])

            # Eval metrics estimation.
            prob = F.softmax(outputs, dim=1).cpu().detach()
            precision, recall, thresholds = precision_recall_curve(lbls[idx].cpu(),
                                                                   prob[idx, 1])
            fscore = (2 * precision * recall) / (precision + recall + 10e-6)  # calculate the f1 score
            epoch_f1 = fscore.max()
            max_f1_index = np.argmax(fscore)
            threshold = thresholds[max_f1_index]
            epoch_pre = precision[max_f1_index]
            epoch_rec = recall[max_f1_index]
            epoch_auc = roc_auc_score(lbls[idx].cpu(), prob[idx, 1])
            epoch_loss = loss.item()
            epoch_acc = accuracy_score(lbls[idx].cpu(), prob[idx, 1] > threshold)

        # Print training information.
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print(f'{phase} Loss: {epoch_loss:.4f}  '
                  # f'Auc : {epoch_auc} '
                  f'Acc: {epoch_acc:.4f} Pre: {epoch_pre:.4f} Rec: {epoch_rec:.4f}')
        if phase == 'val':
            loss_val.append(epoch_loss)
            # Training process visualization.
            if fun_args['tensorboard']:
                writer.add_scalars('loss', {
                    'train_loss': epoch_loss,
                    'eval_loss': epoch_loss
                }, epoch)
                writer.add_scalar('metrics/f1', epoch_f1, epoch)
                writer.add_scalar('metrics/auc', epoch_auc, epoch)
                writer.add_scalar('metrics/acc', epoch_acc, epoch)
                writer.add_pr_curve('pr_curve', lbls[idx].cpu(), prob[idx, 1], epoch)
            # Save result
            if not os.path.exists('result/{}/'.format(epoch)):
                os.makedirs('result/{}/'.format(epoch))
            np.savetxt('result/{}/pre.csv'.format(epoch), precision)
            np.savetxt('result/{}/rec.csv'.format(epoch), recall)
    if fun_args['tensorboard']:
        writer.close()

# if __name__ == "__main__":


train_model(model, criterion, optimizer, schedular, cfg['optim_args']['max_epoch'])

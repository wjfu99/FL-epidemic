import numpy as np
# from modules.models import MultiScaleFedGNN
from modules import MultiScaleFedGNN
# import modules
from Agent_Epi_Sim import Engine
from utils import hypergraph_generator, hypergraph_sequence_generator, label_generator, hypergraph2hyperindex
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

# load the cfg file
# cfg = configparser.ConfigParser()
# cfg.read('cfg.ini')
with open("config.json", 'r') as f:
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
    traj = np.load('../HGNN-Epidemic/bj-sim/privacy/noposterior/trace_array.npy')
    usr_num = traj.shape[0]
    lbls = np.load('../HGNN-Epidemic/bj-sim/privacy/label.npy')
    lbls = torch.tensor(lbls).to(device).squeeze()
elif env_args['dataset'] == 'first_edition':
    traj = eng.get_traj_mat
    usr_num = eng.get_usr_num
    eng.next(env_args['sim_days']*48)
    lbls = eng.get_usr_states
    lbls = torch.tensor(label_generator(lbls)).to(device)
elif env_args['dataset'] == 'large':
    traj = np.load("./Agent_Epi_Sim/data/beijing/processed_data/traj_mat(filled).npy")
    usr_num = traj.shape[0]
    # lbls = joblib.load('./datasets/beijing/large-filled/label')  # TODO the label file should be moved.
    # lbls = label_generator(lbls)
    # generate with old version simulator.
    lbls = np.load('./data/label.npy')
    lbls = torch.tensor(lbls).to(device).squeeze()
elif env_args['dataset'] == 'largec': # chose as the benchmark.
    data_path = './datasets/beijing/large-filled-clustered/'
    traj = np.load(data_path + "traj_mat(filled,sample).npy")
    usr_num = traj.shape[0]
    lbls = np.load(data_path + 'label.npy')
    lbls = torch.tensor(lbls).to(device).squeeze()
    env_args['sim_days'] = 14
elif env_args['dataset'] == 'small':
    # traj = np.load("./Agent_Epi_Sim/data/beijing/processed_data/traj_mat.npy")
    traj = np.load('../HGNN-Epidemic/bj-sim/privacy/noposterior/trace_array.npy')
    # traj = joblib.load('./datasets/beijing/small-unfilled/eng(small)')
    usr_num = traj.shape[0]
    lbls = joblib.load('./datasets/beijing/small-unfilled/label(small)')  # TODO the label file should be moved.
    lbls = label_generator(lbls)
    lbls = torch.tensor(lbls).to(device).squeeze()

train_ratio = cfg['env_args']['train_ratio']
sample_num = lbls.shape[0]
train_num = int(sample_num*train_ratio)
idx_train = np.array(range(train_num))
idx_test = np.array(range(train_num, sample_num))


# Generate the hypergraph sequence
graph_seq, index_seq = hypergraph_sequence_generator(
    traj[:, :env_args['sim_days']*48],
    seq_num=env_args['seq_num'],
    device=device, unique_len=env_args['unique_len'])
# H = np.load('../HGNN-Epidemic/bj-sim/privacy/noposterior/H_un=10_rm01=True.npy')
# graph_seq = [hypergraph2hyperindex(H, device)]

# Prepare region epidemic embedding
if model_args['macro']:
    """
    inputs: shape ()
    return: shape ()
    """
    reg_emb = np.load(data_path + 'region_epi_emb.npy')
    reg_emb = reg_emb.squeeze(0)
    reg_emb = reg_emb.transpose((1, 0, 2))
    emb_seq = []
    emb_unit_len = 48  # 1 day
    reg_emb_dim = reg_emb.shape[-1]
    # TODO without consideration of rnn module. (only one graph is create)
    padding_len = env_args['sim_days'] - reg_emb.shape[1]
    unique_num = env_args['sim_days']*48//env_args['unique_len']//env_args['seq_num']
    for i, index in enumerate(index_seq):
        edge_emb = np.zeros((len(index), reg_emb_dim))
        for j, (loc, t) in enumerate(index):
            day = (i + t/unique_num*env_args['sim_days']/env_args['seq_num'])
            time = (i * unique_num + t) * env_args['unique_len']
            reg_emb_idx = time // emb_unit_len
            if reg_emb_idx < padding_len:
                edge_emb[j, :] = np.zeros(reg_emb_dim)
            else:
                edge_emb[j, :] = reg_emb[loc, reg_emb_idx-padding_len, :]
        edge_emb = torch.tensor(edge_emb).to(device)
        emb_seq.append(edge_emb)
    cfg['model_args']['loc_emb_seq'] = emb_seq
    cfg['model_args']['loc_emb_dim'] = reg_emb_dim
        # edge_emb = np.zeros((len(index), reg_emb.shape[-1]))
        # loc_list = np.array([loc for loc, _ in index])
        # edge_emb = reg_emb[loc_list, :, :]


# Fake location generation 
if model_args['loc_dp']:
    fake_trajs_dir = data_path + 'fake_trajs.npy'
    reg_epi_freq = np.load(data_path + 'region_epi_freq.npy')
    real_locs, fake_locs = plausible_loc_gen(traj[:, :env_args['sim_days']*48], seq_num=env_args['seq_num'],
                                      unique_len=env_args['unique_len'], fake_trajs_dir=fake_trajs_dir,
                                      epi_risk=reg_epi_freq, index_seq=index_seq)
    # real_locs, fake_locs = fake_loc_gen(traj[:, :env_args['sim_days']*48], seq_num=env_args['seq_num'])
    cfg['model_args']['real_locs'], cfg['model_args']['fake_locs'] = real_locs, fake_locs

# Model definition
model = MultiScaleFedGNN(usr_num=usr_num, **cfg['model_args']).to(device)
summary(model)
criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1]).to(device))
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
                outputs = model(graph_seq, epoch, graph_seq)
                loss = criterion(outputs[idx], lbls[idx])

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    if cfg['model_args']['loc_dp']:
                        model_state = model.state_dict() # requires_grad == false with state_dict()
                        user_emb = copy.deepcopy(model_state['usr_emb.weight'])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    # clip on the user embedding.
                    if cfg['model_args']['loc_dp']:
                        new_user_emb = model_state['usr_emb.weight']
                        user_emb_upd = new_user_emb - user_emb
                        user_emb_upd = usr_emb_clip(user_emb_upd, cfg['model_args']['loc_clip'])
                        user_emb = user_emb + user_emb_upd
                        model_state['usr_emb.weight'] = user_emb
                        model.load_state_dict(model_state) # TODO: this code may be redundant.
                        # user_emb = new_user_emb
                    # Add noise on the updated parameters.
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
    if fun_args['tensorboard']:
        writer.close()

# if __name__ == "__main__":


train_model(model, criterion, optimizer, schedular, cfg['optim_args']['max_epoch'])

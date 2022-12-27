import numpy as np
# from modules.models import MultiScaleFedGNN
from modules import MultiScaleFedGNN
# import modules
from Agent_Epi_Sim import eng
from utils import hypergraph_generator, hypergraph_sequence_generator, label_generator
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import os
import configparser
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve

# load the cfg file
# cfg = configparser.ConfigParser()
# cfg.read('cfg.ini')
with open("config.json", 'r') as f:
    cfg = json.load(f)

# For more specific debugging results.
if cfg['fun_args']["debug"]:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cpu")
else:
    # Define the default GPU device
    device = torch.device("cuda:1")


# Process the epidemic data.
# a = eng.get_traj_mat
traj = eng.get_traj_mat
usr_num = eng.get_usr_num
eng.next(20*48)
eng.next(48)
lbls = eng.get_usr_states
lbls = torch.tensor(label_generator(lbls)).to(device)

train_ratio = cfg['env_args']['train_ratio']
sample_num = lbls.shape[0]
train_num = int(sample_num*train_ratio)
idx_train = np.array(range(train_num))
idx_test = np.array(range(train_num, sample_num))


# Generate the hypergraph sequence
graph_seq = hypergraph_sequence_generator(traj[:, :20*48], seq_num=20, device=device)

model = MultiScaleFedGNN(usr_num=usr_num, **cfg).to(device)
summary(model)
criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1]).to(device))
optimizer = optim.Adam(model.parameters(), lr=cfg['optim_args']["lr"], weight_decay=cfg['optim_args']["weight_decay"]) # TODO: SGD for FL?
schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['optim_args']["milestones"], gamma=cfg['optim_args']['gamma'])

writer = SummaryWriter()


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
                outputs = model(graph_seq, graph_seq)
                loss = criterion(outputs[idx], lbls[idx])

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # Eval metrics estimation.
            prob = F.softmax(outputs, dim=1).cpu().detach()
            precision, recall, thresholds = precision_recall_curve(lbls[idx].cpu(),
                                                                   prob[idx, 1])
            fscore = (2 * precision * recall) / (precision + recall)  # calculate the f1 score
            epoch_f1 = fscore.max()
            max_f1_index = np.argmax(epoch_f1)
            threshold = thresholds[max_f1_index]
            epoch_pre = precision[max_f1_index]
            epoch_rec = recall[max_f1_index]
            # epoch_auc = roc_auc_score(lbls[idx].cpu(), prob[idx, 1])
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
            writer.add_scalars('loss', {
                'train_loss': epoch_loss,
                'eval_loss': epoch_loss
            }, epoch)
            writer.add_pr_curve('pr_curve'+str(epoch), lbls[idx].cpu(), prob[idx, 1])
    writer.close()

# if __name__ == "__main__":


train_model(model, criterion, optimizer, schedular, cfg['optim_args']['max_epoch'])

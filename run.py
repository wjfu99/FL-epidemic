import numpy as np
# from modules.models import MultiScaleFedGNN
from modules import MultiScaleFedGNN
# import modules
from Agent_Epi_Sim import eng
from utils import hypergraph_generator, hypergraph_sequence_generator
import torch
import os

# For more specific debugging results.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Process the epidemic data.
# a = eng.get_traj_mat
traj = eng.get_traj_mat
usr_num = eng.get_usr_num
# eng.next(20*48)
# eng.next(48)
label = eng.get_usr_label
idx_train = 1
idx_test = 1

# Define the default GPU device
device = torch.device("cpu")

# Generate the hypergraph sequence
graph_seq = hypergraph_sequence_generator(traj[:, :20*48], seq_num=20, device=device)

model = MultiScaleFedGNN(usr_num=usr_num).to(device)

outputs = model(graph_seq, graph_seq)

def train_model(model, criterion, optimizer, scheduler, num_epochs, print_freq=10):

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')
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
                outputs = model(graph_seq)
                loss = criterion(outputs[idx], label[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

# if __name__ == "__main__":

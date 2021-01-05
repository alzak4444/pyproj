import copy
import csv
import os
import time
import kornia

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs, batch_size):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    global_step = 0

    boardwriter = SummaryWriter(comment=f'testing')

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)

                # Random horizontal flipping / translate / rotate
                if phase == 'Train':
                    transs = []
                    if np.random.uniform(0, 1.0) > 0.5:
                        transs.append(kornia.Hflip())
                    if np.random.uniform(0, 1.0) > 0.5:
                        transs.append(kornia.Vflip())
#                    if np.random.uniform(0, 1.0) > 0.5:
#                        dx = np.random.uniform(-100, 100)
#                        dy = np.random.uniform(-100, 100)
#                        par = torch.tensor([dx,dy])
#                        par = par.cuda()
#                        transs.append(kornia.Translate(par.repeat(batch_size, 1)))
                    if np.random.uniform(0, 1.0) > 0.5:
                        angle = np.random.uniform(0, 3.0)
                        par = torch.tensor(angle)
                        par = par.cuda()
                        transs.append(kornia.Rotate(par.repeat(batch_size)))

                    if len(transs) > 0:
                        tr1 = torch.nn.Sequential(*transs)
                        inputs = tr1(inputs)
                        masks = tr1(masks)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs) # returns a dict
                    out1 = outputs['out']
                    loss = criterion(out1, masks)

#                    y_pred = out1.data.cpu().numpy().ravel()
#                    y_true = masks.data.cpu().numpy().ravel()
#                    for name, metric in metrics.items():
#                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
#                            batchsummary[f'{phase}_{name}'].append(
#                                metric(y_true > 0, y_pred > 0.1))
#                        else:
#                            batchsummary[f'{phase}_{name}'].append(
#                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    global_step += 1
                    if phase == 'Train':
                        boardwriter.add_scalar('Loss/train', loss.item(), global_step)

                        loss.backward()
                        optimizer.step()

#                        if global_step % (4) == 0:
#                            boardwriter.add_images('images', inputs, global_step)
#                            boardwriter.add_images('masks_true', masks, global_step)
#                            boardwriter.add_images('masks_pred', out1, global_step)
                    else:
                        boardwriter.add_scalar('Loss/test', loss.item(), global_step)

            vItem = loss.item()
            batchsummary['epoch'] = epoch
            batchsummary[f'{phase}_loss'] = vItem
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)

        if epoch % 4 == 0 and epoch < num_epochs :
            torch.save(model, bpath / f'weights-{epoch}.pt')

        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import MicrowaveDataset
from resnet import resnet50, resnet101


def train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_index, tb_writer, device):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):

        # Every data instance is an input + label pair
        inputs = data['inputs'].to(device)
        labels = data['labels'].to(device)

        # Zero your gradients for every batch!
        # optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.float(), labels.float())

        # print(loss.values())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        optimizer.zero_grad()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(device):
    full_dataset = MicrowaveDataset('/media/hdd_4tb/Datasets/rohde_and_schwarz_dataset/radar-task/dummy_measurements/volumes',
                                    '/media/hdd_4tb/Datasets/rohde_and_schwarz_dataset/radar-task/dummy_measurements/labels_transformed.json')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)

    loss_fn = torch.nn.MSELoss()

    model = resnet50(num_classes=6)
    # model = resnet101(num_classes=6)
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # scheduler = ExponentialLR(optimizer, end_lr=1e-5, num_iter=50)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=5)

    # learning rate finder
    # lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
    # lr_finder.range_test(training_loader, end_lr=100, num_iter=100)
    # lr_finder.plot()
    # lr_finder.reset()

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/microwave_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 50
    patience = 0
    best_vloss = 1_000_000.

    model.to(device)
    best_model = None
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_number, writer, device)


        # We don't need gradients on to do reporting
        model.train(False)

        model.eval()
        torch.cuda.empty_cache()

        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs = vdata['inputs'].to(device)
                vlabels = vdata['labels'].to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs.float().detach(), vlabels.float().detach())
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            patience = 0
            best_vloss = avg_vloss
            best_model = model.state_dict()
        else:
            patience += 1
        if patience > 8:
            break

        if epoch % 3 == 0:
            scheduler.step()
        epoch_number += 1
        torch.cuda.empty_cache()
    model_path = 'model_{}_{}'.format(timestamp, epoch_number)
    torch.save(best_model, model_path)


if __name__ == '__main__':

    device = torch.device("cpu")
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    train(device)

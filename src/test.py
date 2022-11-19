import torch

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import MicrowaveDataset
from resnet import resnet50



def test(device, model_path):
    test_dataset = MicrowaveDataset('../data/radar_measurements/volumes',
                                    '../data/radar_measurements/labels_transformed.json')

    # Create data loaders for our datasets; 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)

    loss_fn = torch.nn.MSELoss()

    model = resnet50(num_classes=6)
    model.load_state_dict(torch.load(model_path))

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/microwave_tester_{}'.format(timestamp))

    model.to(device)
    model.eval()
    torch.cuda.empty_cache()


    running_tloss = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for i, tdata in enumerate(test_loader):
            tinputs = tdata['inputs'].to(device)
            tlabels = tdata['labels'].to(device)
            toutputs = model(tinputs)

            tloss = loss_fn(toutputs.float().detach(), tlabels.float().detach())
            running_tloss += tloss

             # Gather data and report
            if i % 50 == 49:
                last_loss = running_tloss / 50  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = i + 1
                # Log the running loss averaged per batch
                # for testing
                writer.add_scalar('Loss/test', last_loss, tb_x)
                writer.flush()
                total_loss += running_tloss
                running_tloss = 0.
    print('  Total avg test loss {} '.format(total_loss/ len(test_loader)))


    torch.cuda.empty_cache()


def inference(model, inputs):

    model.to(device)
    model.eval()
    outputs = model(inputs).detach().cpu().numpy()
    
    ret_dict = {'p_pivot' : outputs[:,:3],
                'alpha' : outputs[:,3],
                'beta': outputs[:,4], 
                'gamma': outputs[:,5]}
    return ret_dict

if __name__ == '__main__':

    device = torch.device("cpu")
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    test(device, '../models/model_20221119_115416_25.pth')

import torch

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import MicrowaveDataset
from resnet import resnet50
from dataset import denormalize_input_angles, denormalize_input_pivot, preprocess_input



def test(device, model_path):
    test_dataset = MicrowaveDataset('/media/hdd_4tb/Datasets/rohde_and_schwarz_dataset/radar-task/radar_measurements/volumes',
                                    '/media/hdd_4tb/Datasets/rohde_and_schwarz_dataset/radar-task/radar_measurements/labels_transformed.json')

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


def inference(model, device, input_file_name, input_dir):

    model.to(device)
    model.train(False)
    model.eval()
    preprocessed_input, _ = preprocess_input(input_file_name, input_dir)

    # convert to batch 1 
    preprocessed_input = torch.unsqueeze(preprocessed_input, 0)

    preprocessed_input = preprocessed_input.to(device)
    print(preprocessed_input.get_device(), device)
    
    outputs = model(preprocessed_input).detach().cpu().squeeze(0).numpy()

    
    angles = denormalize_input_angles(outputs[3:])

    ret_dict = {'p_pivot' : denormalize_input_pivot(outputs[:3]),
                'alpha' : angles[0],
                'beta': angles[1], 
                'gamma': angles[2]}
    
    print(ret_dict)
    
    return ret_dict

if __name__ == '__main__':

    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    #test(device, '../models/model_20221119_115416_25.pth')# -> best model loss 0.11286414414644241 
    
    # test(device, '../models/model_20221120_063007_70.pth') #  -> best model loss 0.11286414414644241 
    # test(device, '../models/model_20221120_060513_45.pth') #  0.14772295951843262 
    # test(device, '../models/model_20221120_053908_31.pth') #  0.13020381331443787 
    # test(device, '../models/model_20221120_051036_30.pth') # loss 0.16905595362186432 
    # test(device, '../models/model_20221119_152727_18_best_0.004.pth') #  0.2610507607460022 

    model = resnet50(num_classes=6)
    model.load_state_dict(torch.load('../models/model_20221120_063007_70.pth'))
    ret = inference(model, device, '20221110-134423-525', '/media/hdd_4tb/Datasets/rohde_and_schwarz_dataset/radar-task/radar_measurements/volumes')


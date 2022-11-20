import cv2
import numpy as np
from resnet import resnet50
from test import inference_radar
from main import process_display

import torch

from datetime import datetime
from dataset import MicrowaveDataset


set_test = False
def compare_values(reference_dict, output_dict):
    
    p_pivot_error = np.sum(np.absolute(reference_dict['p_pivot'] - output_dict['p_pivot']))
    alpha_error = np.absolute(reference_dict['alpha'] - output_dict['alpha'])
    beta_error = np.absolute(reference_dict['beta'] - output_dict['beta'])
    gamma_error = np.absolute(reference_dict['gamma'] - output_dict['gamma'])

    return p_pivot_error, alpha_error, beta_error, gamma_error

# button dimensions (y1,y2,x1,x2)
button = [20,60,50,670]

# function that handles the mousclicks
def process_click(event, x, y,flags, params):
    global set_test
    # check if the click is within the dimensions of the button
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > button[0] and y < button[1] and x > button[2] and x < button[3]:   
            set_test = True
            print('Clicked on Button!')


def resizeAndPad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

if __name__ == '__main__':

    cv2.namedWindow('Radar Detectron')
    cv2.setMouseCallback('Radar Detectron',process_click)
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (720,800))

    # create button image
    control_image = np.zeros((800,720,3), np.uint8)
    control_image[button[0]:button[1],button[2]:button[3]] = 180
    cv2.putText(control_image, 'Load Test Volume',(230,50),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)

    ref_image_name = "ref"
    ref_image_dir = "../examples/"
    ref_image = process_display(
        ref_image_name, ref_image_dir, title = 'Reference'
        #r"20221119-150759-488_reco", r"/media/hdd_4tb/Datasets/rohde_and_schwarz_measurements/"
        )
    ref_dict = {"p_pivot": np.array([
            0.0030615681489981963,
            0.0050230819560664556,
            0.20239102849378485
        ]),
        "alpha": -0.2707527075,
        "beta": -3.068530685,
        "gamma": -2.0847849275}
    
    ref_image = resizeAndPad(ref_image,(300,600))
    
    control_image[70:370,60:660,:] = ref_image[:,:,::-1]

    test_drawn = False
    while True:
        cv2.imshow('Radar Detectron', control_image)
        key = cv2.waitKey(1) & 0xFF
        print("test")
        if test_drawn is False and set_test:
            model = resnet50(num_classes=6)
            model.load_state_dict(torch.load('../models/model_20221120_063007_70.pth'))

            device = torch.device("cpu")
            if torch.cuda.is_available():
                # device = torch.cuda.current_device()
                device = 'cuda'

            test_dict = inference_radar(model, device, '20221110-134426-875', '/media/hdd_4tb/Datasets/rohde_and_schwarz_dataset/radar-task/radar_measurements/volumes')
            test_vol = ['20221110-134426-875', '/media/hdd_4tb/Datasets/rohde_and_schwarz_dataset/radar-task/radar_measurements/volumes']
            test_image = process_display(
            test_vol[0], test_vol[1], title = 'Test image'
            #r"20221119-150759-488_reco", r"/media/hdd_4tb/Datasets/rohde_and_schwarz_measurements/"
            )
            test_image = resizeAndPad(test_image,(300,600))
            control_image[470:770,60:660,:] = test_image[:,:,::-1]
            p_pivot_error, alpha_error, beta_error, gamma_error = compare_values(ref_dict, test_dict)
            cv2.putText(control_image, 'Mislignments :p_pivot {:.4f}, alpha {:.4f}, beta {:.4f}, gamma {:.4f}'.format(p_pivot_error, alpha_error, beta_error, gamma_error),(50,400),cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255),1)

            test_drawn = True

        # KEYBOARD INTERACTIONS
        if key == ord('q'):
            break

        elif key == ord('s'):
            # save the image as such
            #cv2.imwrite('mimi_colour.jpg', img)
            break
        out.write(control_image)
    out.release()
    cv2.destroyAllWindows()
            
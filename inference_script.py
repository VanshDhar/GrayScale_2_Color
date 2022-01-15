import torchvision.transforms as T
import torch
import numpy as np
import os
import sys
import cv2

def infer(img_path,model_path):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_transform = T.Compose([T.ToTensor(),
                                 T.Resize(size=(256,256)),
                                 T.Grayscale(num_output_channels=1),
                                 T.Normalize((0.5), (0.5))
                                 ])
    # Use this on target images(colorful ones)
    #target_transform = T.Compose([T.ToTensor(),
    #                              T.Resize(size=(256,256)),
    #                              T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    label = cv2.imread(img_path)
    label = np.array(label)
    
    image = input_transform(label)
    
    model = torch.load(model_path, map_location=torch.device(device))
    #model.set_params(device=device)
    model.eval()
    #with torch.no_grad():
    
    
    if device == 'cuda':
        image = image.cuda()
    pred_image = model(image.unsqueeze(0))
    
    pred_image = pred_image.squeeze(0)
    #print(pred_image)
    #print(pred_image.size())
    MEAN = 255 * torch.tensor([0.5, 0.5, 0.5],device=device)
    STD = 255 * torch.tensor([0.5, 0.5, 0.5],device=device)
    #pred_image = pred_image.mul_(255*0.5).add_(255*0.5)
    pred_image = pred_image * STD[:, None, None] + MEAN[:, None, None]
    #pred_image = pred_image.sub_(torch.min(pred_image)).div_(torch.max(pred_image) - torch.min(pred_image))
    pred_image = pred_image.cpu().detach().numpy()
    
    #pred_image = (pred_image - np.min(pred_image)) / (np.max(pred_image) - np.min(pred_image))
    #pred_image *= 255
    
    pred_image = pred_image.transpose((1, 2, 0))
    #pred_image = (pred_image*0.5)+0.5
    cv2.imwrite('./Predicted_image.jpg', pred_image)
    
    
    
    #label = self.target_transform(label)

if __name__ == "__main__":  

    img_path = sys.argv[1]#"/Users/vanshdhar/Desktop/samsung_challenge/landscape_images/733.jpg"# sys.argv[1]
    model_path = sys.argv[2]#"/Users/vanshdhar/Desktop/samsung_challenge/saved_model/nn_e016.pt" #sys.argv[2]


    if os.path.exists(img_path) and os.path.exists(model_path):
        infer(img_path,model_path)
    elif not os.path.exists(img_path):
        print("Image path doesn't exists")
    elif not os.path.exists(model_path):
        print("Model path doesn't exists")
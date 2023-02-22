import cv2
import torch
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

transform= transforms.Compose([
        transforms.ToTensor(), 
    ])
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

cv2.namedWindow("camera",cv2.WINDOW_NORMAL)
cv2.resizeWindow("camera",300,300)

def add_bbox(img_tensor,annotation):
    img=img_tensor[0]
    img = img.cpu().data
    img=img.permute(1, 2, 0)#(c,w,h)->(w,h,c)
    img=img.detach().numpy()
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    for box,label,score in zip(annotation[0]['boxes'],annotation[0]['labels'],annotation[0]['scores']):
        if label==1 and score>0.8:
            box=box.cpu().detach().numpy()
            xmin, ymin, xmax, ymax = box

            # Create a Rectangle patch
            img = cv2.rectangle(img,((int(xmin),int(ymin))),((int(xmax),int(ymax))),(0,0,255),2)
    return img
video=cv2.VideoCapture(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(3)
model.load_state_dict(torch.load("C:/Users/90761/Desktop/kaggle/face-mask-detection/model.pt"))
model.eval()
model.to(device)
while video.isOpened():
    ret,frame=video.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=transform(frame).cuda()#to cuda
    frame=torch.unsqueeze(frame,0)#add one dim
    preds = model(frame)
    frame=add_bbox(frame,preds)
    cv2.imshow("camera",frame)

    key = cv2.waitKey(10)
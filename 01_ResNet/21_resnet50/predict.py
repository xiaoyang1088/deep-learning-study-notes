import argparse
import torch
from PIL import Image
from data import build_transform
from model import resnet50

def predict(checkpoint_path,image_path,device):
    checkpoint=torch.load(
        checkpoint_path,
        map_location='cpu',
    )

    classes=checkpoint['classes']
    model=resnet50(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with Image.open(image_path)as source:
        image=source.convert('RGB')

    transform=build_transform(False)
    image_tensor=transform(image)
    images=image_tensor.unsqueeze(0)    #增加一个维度
    images=images.to(device)

    with torch.no_grad():
        probabilities=model(images).softmax(dim=1)[0]

    values,indices=probabilities.topk(min(5,len(classes)))

    for probability,index in zip(values.tolist(),indices.tolist()):
        print(f"类别：{classes[index]['name']},模型概率：{probability:.2%}")


def main():
    parser=argparse.ArgumentParser(description='用训练的ResNet50预测图片')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    args=parser.parse_args()
    device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
    predict(args.checkpoint,args.image,device)

if __name__=='__main__':
    main()
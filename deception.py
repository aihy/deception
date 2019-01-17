import json

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms

import dataloader
PPATH = "dd/"
def deception_test(pinput):
    with open("dec_data.csv","w") as fp:
        fp.write("path,label\n")
        for item in pinput:
            fp.write(PPATH+item+",0\n")
    model = models.resnet34(pretrained=False, num_classes=2)
    model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load("model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = dataloader.XixiDataset("dec_data.csv",
                                         transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)
    model.eval()
    dec_p = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            output = model(input)
            dec_p.append(F.softmax(output, dim=1).squeeze().tolist()[0])
    
    with open("deception.json","w") as fp:
        json.dump(dec_p,fp)

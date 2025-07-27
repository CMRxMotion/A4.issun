import numpy as np
import torch
import os
import random
from model import IQC
import nibabel as nib
import csv
def normalization(array):
    maxValue = np.percentile(array, 98, interpolation='nearest')
    minValue = np.percentile(array, 2, interpolation='nearest')
    re_imgArray = (array - minValue) / (maxValue - minValue + 0.000000001)
    return re_imgArray

def crop_or_pad_slice_to_size(img, nx, ny):

    x, y,z = img.shape
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny,:]
    else:
        slice_cropped = np.zeros((nx, ny,z))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :,:] = img[:, y_s:y_s + ny,:]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y,:] = img[x_s:x_s + nx, :,:]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y,:] = img[:, :,:]
    return slice_cropped

def readImg(imgPath):
    imgReader = nib.load(imgPath)
    img=imgReader.dataobj
    return img[...,0]


def testCase(img,model):


    pre = []
    for r in range(50):
        z = np.shape(img)[0]
        fixedSlice=[2,6]
        channel_rand_array = random.sample([0,1,4,2,6], 3)
        channel_rand_array = fixedSlice+channel_rand_array

        channel_rand_array = [min(x, z-1) for x in channel_rand_array]

        img7 = img[channel_rand_array]
        img7 = np.expand_dims(img7, axis=0)

        fixedSlice = [2, 6]
        channel_rand_array = random.sample([0,1,4,3,5], 3)
        channel_rand_array = fixedSlice+channel_rand_array

        channel_rand_array = [min(x, z-1) for x in channel_rand_array]

        img5 = img[channel_rand_array]
        img5 = np.expand_dims(img5, axis=0)

        img7 = torch.from_numpy(img7)
        img5 = torch.from_numpy(img5)

        img7 = img7.to(device=device, dtype=torch.float32)
        img5 = img5.to(device=device, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            label_pred = model(img5, img7)

        predOut = torch.argmax(label_pred)
        predOut = predOut.item()
        pre.append(predOut)

    zero = pre.count(0)
    one = pre.count(1)
    two = pre.count(2)
    countList=[zero,one,two]
    return countList

def decision(numList):
    first,second,third = numList
    if third+second>70:
        if third>=13 or second-third<5:
            label=2
        else:
            label=1
    else:
        label=0

    if third<13:
        newsecond=second+third*5
    else:
        newsecond=second

    if label==0 :
        if newsecond>=66:
            label=1
    if third>=13:
        label=2

    return label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelCkpts = ["./ckpt/MultiResNetGEFold1_55FT.pth",
            "./ckpt/MultiResNetGEFold2_55FT.pth",
            "./ckpt/MultiResNetGEFold3_55FT.pth",
            "./ckpt/MultiResNetGEFold4_55FT.pth"]

def test(inputPath,outputPath):
    caseDict={}
    inputfileList=os.listdir(inputPath)
    inputfileList=sorted(inputfileList)
    for pt in modelCkpts:
        tmp_model=IQC(3)
        tmp_model.load_state_dict(torch.load(pt))
        tmp_model.to(device=device)
        for f in inputfileList:
            caseName=f.split('.')[0]
            filePath=os.path.join(inputPath,f)
            imgStack=readImg(filePath)
            # imgStack=np.load(filePath)
            crop_imgStack=crop_or_pad_slice_to_size(imgStack,192, 192)
            crop_imgStackNor = np.transpose(crop_imgStack, (-1, 1, 0))
            crop_imgStackNor = normalization(crop_imgStackNor)
            testImg = crop_imgStackNor[1:8]
            preList = []
            for r in range(5):
                pred = testCase(testImg, tmp_model)
                preList.append(pred)
            preList = np.array(preList)
            if caseName not in caseDict:
                caseDict[caseName] = preList
            else:
                caseDict[caseName] += preList
            print(pt,caseName,preList)

    for key in caseDict:
        value = caseDict[key]
        voted = []
        for di in range(value.shape[0]):
            predLabel = decision(value[di])+1
            voted.append(predLabel)
        edpreCountList = [voted.count(1), voted.count(2), voted.count(3)]
        prediction = edpreCountList.index(np.max(edpreCountList))+1
        caseDict[key] = prediction

        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        with open(os.path.join(outputPath,"output.csv"),'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Label'])
            for key in caseDict:
                writer.writerow([key, caseDict[key]])

# validationPath="/exports/lkeb-hpc/xsun/DATA/CMRxMotion/validationTest/"
# output="/exports/lkeb-hpc/xsun/DATA/CMRxMotion/validation/"
# test(validationPath,output)


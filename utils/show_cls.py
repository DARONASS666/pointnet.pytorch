from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet'))
print(sys.path)
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')


opt = parser.parse_args()
print(opt)

test_dataset = ShapeNetDataset(
    root=BASE_DIR + '/shapenetcore_partanno_segmentation_benchmark_v0',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# add a for loop here for each file in the directory


# Directory path
directory = opt.model

files = os.listdir(directory)

# Filter files that match a specific pattern (e.g., cls_model_*.pth)
files = [file for file in files if file.startswith('cls_model_') and file.endswith('.pth')]

# Sort the files by their names
sorted_files = sorted(files, key=lambda x: (int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else float('inf'), x))

overall_losses = []
overall_accuracies = []
# Iterate over sorted files
for filename in sorted_files:
    full_path = os.path.join(directory, filename)
    # Process individual file
    print(filename, full_path)

    classifier = PointNetCls(k=len(test_dataset.classes))
    classifier.to(device)
    classifier.load_state_dict(torch.load(full_path), strict=False)
    classifier.eval()

    losses = []
    accuracies = []
    for i, data in enumerate(testdataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target[:, 0])
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        pred, _, _ = classifier(points)
        loss = F.nll_loss(pred, target)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        tmp_loss = loss.data.item()
        tmp_acc = correct / float(32)
        losses.append(tmp_loss)
        accuracies.append(tmp_acc)
        # print('i:%d  loss: %f accuracy: %f' % (i, tmp_loss, tmp_acc))
        # print("Avg Loss:", np.mean(losses))
        # print("Avg Accuracy:", np.mean(accuracies))

    # print("Avg Loss:", filename, np.mean(losses))
    # print("Avg Accuracy:", filename, np.mean(accuracies))

    overall_losses.append((int(filename.split('_')[-1].split('.')[0]), np.mean(losses)))
    overall_accuracies.append((int(filename.split('_')[-1].split('.')[0]), np.mean(accuracies)))

print("Losses:", overall_losses)
print("==============================================")
print("Accuracies:", overall_accuracies)

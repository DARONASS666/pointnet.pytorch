#  use --dataset "/Users/daron/Desktop/daron/GEORGIATECH/COURSES/FALL2023/CS7643/final_project/point-pytorch/pointnet.pytorch/ModelNet40_numpy" --dataset_type "modelnet40"
# or use --dataset "/Users/daron/Desktop/daron/GEORGIATECH/COURSES/FALL2023/CS7643/final_project/point-pytorch/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0" --dataset_type "shapenet"
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys

# print(sys.path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet'))
print(sys.path)
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    print("Cuda Available: ", torch.cuda.is_available(), " | current device: ", device)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='train')

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.to(device)

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            if opt.dataset_type == 'shapenet':
                target = target[:, 0]
                points = points.transpose(2, 1)
            elif opt.dataset_type == 'modelnet40':
                target = target[:, 0]
                points = points.float().transpose(2, 1)
            points, target = points.to(device), target.to(device)
            # points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            if i % 100 == 0:
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            if i % 100 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                if opt.dataset_type == 'shapenet':
                    target = target[:, 0]
                    points = points.transpose(2, 1)
                elif opt.dataset_type == 'modelnet40':
                    target = target[:, 0]
                    points = points.float().transpose(2, 1)
                points, target = points.to(device), target.to(device)
                # points, target = points, target
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))
        scheduler.step()
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        if opt.dataset_type == 'shapenet':
            target = target[:, 0]
            points = points.transpose(2, 1)
        elif opt.dataset_type == 'modelnet40':
            target = target[:, 0]
            points = points.float().transpose(2, 1)
        points, target = points.to(device), target.to(device)
        # points, target = points, target
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))


if __name__ == '__main__':
    train()

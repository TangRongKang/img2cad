"""
CubiCasa5k 模型训练脚本
用于训练户型图分割模型（房间、图标、热图）
"""
import matplotlib
matplotlib.use('pdf')  # 使用PDF后端，适合服务器环境
import sys
import os
import logging
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
# 数据增强相关
from floortrans.loaders.augmentations import (RandomCropToSizeTorch,
                                              ResizePaddedTorch,
                                              Compose,
                                              DictToTensor,
                                              ColorJitterTorch,
                                              RandomRotations)
from torchvision.transforms import RandomChoice
from torch.utils import data
from torch.nn.functional import softmax
from tqdm import tqdm

# 数据加载、模型、损失函数、评估指标
from floortrans.loaders import FloorplanSVG
from floortrans.models import get_model
from floortrans.losses import UncertaintyLoss
from floortrans.metrics import get_px_acc, runningScore
from tensorboardX import SummaryWriter  # TensorBoard日志记录
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


def train(args, log_dir, writer, logger):
    """
    主训练函数
    
    Args:
        args: 命令行参数
        log_dir: 日志目录路径
        writer: TensorBoard写入器
        logger: 日志记录器
    """
    # 保存训练参数到JSON文件
    with open(log_dir+'/args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    # ========== 数据增强设置 ==========
    # 根据是否启用缩放，选择不同的数据增强策略
    if args.scale:
        # 启用缩放：随机选择裁剪或缩放填充
        aug = Compose([RandomChoice([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                                     ResizePaddedTorch((0, 0), data_format='dict', size=(args.image_size, args.image_size))]),
                       RandomRotations(format='cubi'),  # 随机旋转
                       DictToTensor(),  # 转换为张量
                       ColorJitterTorch()])  # 颜色抖动
    else:
        # 不启用缩放：只使用裁剪
        aug = Compose([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                       RandomRotations(format='cubi'),
                       DictToTensor(),
                       ColorJitterTorch()])

    # ========== 数据加载器设置 ==========
    writer.add_text('parameters', str(vars(args)))  # 将参数写入TensorBoard
    logging.info('Loading data...')
    
    # 创建训练集和验证集
    # FloorplanSVG: 从LMDB数据库加载SVG标注的户型图数据
    train_set = FloorplanSVG(args.data_path, 'train.txt', format='lmdb',
                             augmentations=aug)  # 训练集使用数据增强
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                           augmentations=DictToTensor())  # 验证集只转换为张量，不使用增强

    # 设置数据加载的工作进程数
    if args.debug:
        num_workers = 0  # 调试模式：单进程，便于调试
        print("In debug mode.")
        logger.info('In debug mode.')
    else:
        num_workers = 8  # 正常模式：8个进程并行加载数据

    # 创建数据加载器
    trainloader = data.DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=num_workers, shuffle=True, pin_memory=True)
    valloader = data.DataLoader(val_set, batch_size=1,  # 验证时batch_size=1
                                num_workers=num_workers, pin_memory=True)

    # ========== 模型设置 ==========
    logging.info('Loading model...')
    # 输出通道划分：[热图(21), 房间分割(12), 图标分割(11)] = 44个类别
    input_slice = [21, 12, 11]
    
    if args.arch == 'hg_furukawa_original':
        # 使用Hourglass网络架构
        model = get_model(args.arch, 51)  # 先加载51类的预训练模型
        criterion = UncertaintyLoss(input_slice=input_slice)  # 不确定性损失函数
        
        # 如果提供了预训练权重，加载Furukawa模型的权重
        if args.furukawa_weights:
            logger.info("Loading furukawa model weights from checkpoint '{}'".format(args.furukawa_weights))
            checkpoint = torch.load(args.furukawa_weights)
            model.load_state_dict(checkpoint['model_state'])
            criterion.load_state_dict(checkpoint['criterion_state'])

        # 修改最后的卷积层和上采样层，适配新的类别数
        model.conv4_ = torch.nn.Conv2d(256, args.n_classes, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(args.n_classes, args.n_classes, kernel_size=4, stride=4)
        # 使用Kaiming初始化新层的权重
        for m in [model.conv4_, model.upsample]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    else:
        # 其他架构：直接创建指定类别数的模型
        model = get_model(args.arch, args.n_classes)
        criterion = UncertaintyLoss(input_slice=input_slice)

    model.cuda()  # 将模型移到GPU

    # 为TensorBoard绘制模型计算图
    dummy = torch.zeros((2, 3, args.image_size, args.image_size)).cuda()
    model(dummy)  # 前向传播一次以构建计算图
    writer.add_graph(model, dummy)

    # ========== 优化器和学习率调度器设置 ==========
    # 设置优化参数：模型参数和损失函数参数使用相同的学习率
    params = [{'params': model.parameters(), 'lr': args.l_rate},
              {'params': criterion.parameters(), 'lr': args.l_rate}]
    
    if args.optimizer == 'adam-patience':
        # Adam优化器 + 基于验证损失的自动学习率衰减
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=0.5)
    elif args.optimizer == 'adam-patience-previous-best':
        # Adam优化器 + 基于最佳模型的学习率衰减（手动控制）
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        # 注意：这个模式下scheduler在训练循环中手动控制
    elif args.optimizer == 'sgd':
        # SGD优化器 + 自定义学习率衰减函数
        def lr_drop(epoch):
            return (1 - epoch/args.n_epoch)**0.9  # 线性衰减
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=10**-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop)
    elif args.optimizer == 'adam-scheduler':
        # Adam优化器 + 阶梯式学习率衰减
        def lr_drop(epoch):
            return 0.5 ** np.floor(epoch / args.l_rate_drop)  # 每l_rate_drop个epoch减半
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop)

    # ========== 训练状态初始化 ==========
    first_best = True  # 标记是否是第一次达到最佳结果
    best_loss = np.inf  # 最佳验证损失（不含方差）
    best_loss_var = np.inf  # 最佳验证损失（含方差）
    best_train_loss = np.inf  # 最佳训练损失
    best_acc = 0  # 最佳像素准确率
    start_epoch = 0  # 起始epoch
    
    # 初始化评估指标计算器
    running_metrics_room_val = runningScore(input_slice[1])  # 房间分割指标（12类）
    running_metrics_icon_val = runningScore(input_slice[2])  # 图标分割指标（11类）
    
    # 用于学习率调度的变量
    best_val_loss_variance = np.inf
    no_improvement = 0  # 连续无改善的epoch数
    
    # ========== 加载检查点（如果提供） ==========
    if args.weights is not None:
        if os.path.exists(args.weights):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model_state'])
            criterion.load_state_dict(checkpoint['criterion_state'])
            if not args.new_hyperparams:
                # 如果使用旧的超参数，也加载优化器状态
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                logger.info("Using old optimizer state.")
            logger.info("Loaded checkpoint '{}' (epoch {})".format(args.weights, checkpoint['epoch']))
        else:
            logger.info("No checkpoint found at '{}'".format(args.weights)) 

    # ========== 训练循环 ==========
    for epoch in range(start_epoch, args.n_epoch):
        model.train()  # 设置为训练模式
        lossess = []  # 存储每个batch的总损失
        losses = pd.DataFrame()  # 存储详细损失（字典形式）
        variances = pd.DataFrame()  # 存储方差
        ss = pd.DataFrame()  # 存储s值（不确定性参数）
        
        # ========== 训练阶段 ==========
        for i, samples in tqdm(enumerate(trainloader), total=len(trainloader),
                               ncols=80, leave=False):
            # 将数据移到GPU
            images = samples['image'].cuda(non_blocking=True)
            labels = samples['label'].cuda(non_blocking=True)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)
            lossess.append(loss.item())
            # 记录详细损失信息
            losses = losses.append(criterion.get_loss(), ignore_index=True)
            variances = variances.append(criterion.get_var(), ignore_index=True)
            ss = ss.append(criterion.get_s(), ignore_index=True)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算平均损失
        avg_loss = np.mean(lossess)
        avg_loss = np.inf  # 注意：这里被设置为inf，可能是bug，但保留原代码
        loss = losses.mean()  # 平均详细损失（字典）
        variance = variances.mean()  # 平均方差
        s = ss.mean()  # 平均s值

        logging.info("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, avg_loss))

        # 记录训练指标到TensorBoard
        writer.add_scalars('training/loss', loss, global_step=1+epoch)
        writer.add_scalars('training/variance', variance, global_step=1+epoch)
        writer.add_scalars('training/s', s, global_step=1+epoch)
        current_lr = {'base': optimizer.param_groups[0]['lr'],
                      'var': optimizer.param_groups[1]['lr']}
        writer.add_scalars('training/lr', current_lr, global_step=1+epoch)

        # ========== 验证阶段 ==========
        model.eval()  # 设置为评估模式
        val_losses = pd.DataFrame()
        val_variances = pd.DataFrame()
        val_ss = pd.DataFrame()
        px_rooms = 0  # 房间像素准确率累计
        px_icons = 0  # 图标像素准确率累计
        total_px = 0  # 总像素数
        
        for i_val, samples_val in tqdm(enumerate(valloader), total=len(valloader), ncols=80, leave=False):
            with torch.no_grad():  # 验证时不需要计算梯度
                images_val = samples_val['image'].cuda(non_blocking=True)
                labels_val = samples_val['label'].cuda(non_blocking=True)

                # 前向传播
                outputs = model(images_val)
                # 将标签插值到模型输出尺寸（如果尺寸不匹配）
                labels_val = F.interpolate(labels_val, size=outputs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, labels_val)

                # 提取房间分割预测和真实值
                # outputs形状: (1, 44, H, W)，其中[21:33]是房间分割logits
                room_pred = outputs[0, input_slice[0]:input_slice[0]+input_slice[1]].argmax(0).data.cpu().numpy()
                room_gt = labels_val[0, input_slice[0]].data.cpu().numpy()
                running_metrics_room_val.update(room_gt, room_pred)  # 更新房间分割指标

                # 提取图标分割预测和真实值
                # outputs[33:44]是图标分割logits
                icon_pred = outputs[0, input_slice[0]+input_slice[1]:].argmax(0).data.cpu().numpy()
                icon_gt = labels_val[0, input_slice[0]+1].data.cpu().numpy()
                running_metrics_icon_val.update(icon_gt, icon_pred)  # 更新图标分割指标
                
                total_px += outputs[0, 0].numel()
                # 计算像素准确率
                pr, pi = get_px_acc(outputs[0], labels_val[0], input_slice, 0)
                px_rooms += float(pr)
                px_icons += float(pi)

                # 记录验证损失
                val_losses = val_losses.append(criterion.get_loss(), ignore_index=True)
                val_variances = val_variances.append(criterion.get_var(), ignore_index=True)
                val_ss = val_ss.append(criterion.get_s(), ignore_index=True)

        # 计算平均验证损失
        val_loss = val_losses.mean()
        val_variance = val_variances.mean()
        logging.info("val_loss: "+str(val_loss))
        writer.add_scalars('validation loss', val_loss, global_step=1+epoch)
        writer.add_scalars('validation variance', val_variance, global_step=1+epoch)
        
        # ========== 学习率调度 ==========
        if args.optimizer == 'adam-patience':
            # 基于验证损失自动调整学习率
            scheduler.step(val_loss['total loss with variance'])
        elif args.optimizer == 'adam-patience-previous-best':
            # 手动控制：如果验证损失改善，重置计数器；否则增加计数器
            if best_val_loss_variance > val_loss['total loss with variance']:
                best_val_loss_variance = val_loss['total loss with variance']
                no_improvement = 0
            else:
                no_improvement += 1
            # 如果连续patience个epoch无改善，加载最佳模型并降低学习率
            if no_improvement >= args.patience:
                logger.info("No improvement for " + str(no_improvement) + " epochs, loading last best model and reducing learning rate.")
                checkpoint = torch.load(log_dir+"/model_best_val_loss_var.pkl")
                model.load_state_dict(checkpoint['model_state'])
                # 将学习率降低到原来的10%
                for i, p in enumerate(optimizer.param_groups):
                    optimizer.param_groups[i]['lr'] = p['lr'] * 0.1
                no_improvement = 0

        elif args.optimizer == 'sgd' or args.optimizer == 'adam-scheduler':
            # SGD和Adam-scheduler：按epoch数调整学习率
            scheduler.step(epoch+1)

        # 计算最终验证指标
        val_variance = val_variances.mean()
        val_s = val_ss.mean()
        logger.info("val_loss: "+str(val_loss))
        
        # ========== 计算并记录房间分割指标 ==========
        room_score, room_class_iou = running_metrics_room_val.get_scores()
        writer.add_scalars('validation/room/general', room_score, global_step=1+epoch)
        writer.add_scalars('validation/room/IoU', room_class_iou['Class IoU'], global_step=1+epoch)
        writer.add_scalars('validation/room/Acc', room_class_iou['Class Acc'], global_step=1+epoch)
        running_metrics_room_val.reset()  # 重置指标，准备下一轮

        # ========== 计算并记录图标分割指标 ==========
        icon_score, icon_class_iou = running_metrics_icon_val.get_scores()
        writer.add_scalars('validation/icon/general', icon_score, global_step=1+epoch)
        writer.add_scalars('validation/icon/IoU', icon_class_iou['Class IoU'], global_step=1+epoch)
        writer.add_scalars('validation/icon/Acc', icon_class_iou['Class Acc'], global_step=1+epoch)
        running_metrics_icon_val.reset()

        # 记录验证损失到TensorBoard
        writer.add_scalars('validation/loss', val_loss, global_step=1+epoch)
        writer.add_scalars('validation/variance', val_variance, global_step=1+epoch)
        writer.add_scalars('validation/s', val_s, global_step=1+epoch)

        # ========== 保存最佳模型 ==========
        # 保存最佳验证损失（含方差）的模型
        if val_loss['total loss with variance'] < best_loss_var:
            best_loss_var = val_loss['total loss with variance']
            logger.info("Best validation loss with variance found saving model...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'best_loss': best_loss}
            torch.save(state, log_dir+"/model_best_val_loss_var.pkl")
            
            # ========== 在TensorBoard中绘制示例预测图像 ==========
            if args.plot_samples:
                # 只处理前4个验证样本
                for i, samples_val in enumerate(valloader):
                    with torch.no_grad():
                        if i == 4:
                            break

                        images_val = samples_val['image'].cuda(non_blocking=True)
                        labels_val = samples_val['label'].cuda(non_blocking=True)

                        # 第一次达到最佳时，保存输入图像和标签
                        if first_best:
                            # 保存输入图像
                            writer.add_image("Image "+str(i), images_val[0])
                            # 保存所有标签通道（热图、房间、图标）
                            for j, l in enumerate(labels_val.squeeze().cpu().data.numpy()):
                                fig = plt.figure(figsize=(18, 12))
                                plot = fig.add_subplot(111)
                                if j < 21:
                                    # 前21个通道是热图，值域[0,1]
                                    cax = plot.imshow(l, vmin=0, vmax=1)
                                else:
                                    # 后面的通道是分割图，值域[0,19]
                                    cax = plot.imshow(l, vmin=0, vmax=19, cmap=plt.cm.tab20)
                                fig.colorbar(cax)
                                writer.add_figure("Image "+str(i)+" label/Channel "+str(j), fig)

                        # 模型预测
                        outputs = model(images_val)

                        # 分割输出为热图、房间分割、图标分割
                        pred_arr = torch.split(outputs, input_slice, 1)
                        heatmap_pred, rooms_pred, icons_pred = pred_arr

                        # 对分割输出应用softmax得到概率
                        rooms_pred = softmax(rooms_pred, 1).cpu().data.numpy()
                        icons_pred = softmax(icons_pred, 1).cpu().data.numpy()

                        label = "Image "+str(i)+" prediction/Channel "

                        # 绘制热图预测（21个通道）
                        for j, l in enumerate(np.squeeze(heatmap_pred)):
                            fig = plt.figure(figsize=(18, 12))
                            plot = fig.add_subplot(111)
                            cax = plot.imshow(l, vmin=0, vmax=1)
                            fig.colorbar(cax)
                            writer.add_figure(label+str(j), fig, global_step=1+epoch)

                        # 绘制房间分割预测（argmax结果）
                        fig = plt.figure(figsize=(18, 12))
                        plot = fig.add_subplot(111)
                        cax = plot.imshow(np.argmax(np.squeeze(rooms_pred), axis=0), vmin=0, vmax=19, cmap=plt.cm.tab20)
                        fig.colorbar(cax)
                        writer.add_figure(label+str(j+1), fig, global_step=1+epoch)

                        # 绘制图标分割预测（argmax结果）
                        fig = plt.figure(figsize=(18, 12))
                        plot = fig.add_subplot(111)
                        cax = plot.imshow(np.argmax(np.squeeze(icons_pred), axis=0), vmin=0, vmax=19, cmap=plt.cm.tab20)
                        fig.colorbar(cax)
                        writer.add_figure(label+str(j+2), fig, global_step=1+epoch)

            first_best = False  # 标记已保存过示例图像

        # 保存最佳验证损失（不含方差）的模型
        if val_loss['total loss'] < best_loss:
            best_loss = val_loss['total loss']
            logger.info("Best validation loss found saving model...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'best_loss': best_loss}
            torch.save(state, log_dir+"/model_best_val_loss.pkl")

        # 保存最佳像素准确率的模型
        px_acc = room_score["Mean Acc"] + icon_score["Mean Acc"]
        if px_acc > best_acc:
            best_acc = px_acc
            logger.info("Best validation pixel accuracy found saving model...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, log_dir+"/model_best_val_acc.pkl")

        # 保存最佳训练损失的模型
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            logger.info("Best training loss with variance...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, log_dir+"/model_best_train_loss_var.pkl")

    # ========== 训练结束，保存最终模型 ==========
    logger.info("Last epoch done saving final model...")
    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'criterion_state': criterion.state_dict(),
             'optimizer_state': optimizer.state_dict()}
    torch.save(state, log_dir+"/model_last_epoch.pkl")


if __name__ == '__main__':
    # ========== 参数解析 ==========
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='CubiCasa5k 模型训练参数')
    
    # 模型架构相关
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='模型架构名称')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='模型输出类别数（44 = 21热图 + 12房间 + 11图标）')
    
    # 优化器相关
    parser.add_argument('--optimizer', nargs='?', type=str, default='adam-patience-previous-best',
                        help='优化器类型: adam-patience, adam-patience-previous-best, sgd, adam-scheduler')
    parser.add_argument('--l-rate', nargs='?', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--l-rate-var', nargs='?', type=float, default=1e-3,
                        help='方差参数的学习率')
    parser.add_argument('--l-rate-drop', nargs='?', type=float, default=200,
                        help='学习率衰减的epoch间隔（用于adam-scheduler）')
    parser.add_argument('--patience', nargs='?', type=int, default=10,
                        help='学习率衰减的耐心值（连续无改善的epoch数）')
    
    # 数据相关
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='数据目录路径')
    parser.add_argument('--batch-size', nargs='?', type=int, default=26,
                        help='批次大小')
    parser.add_argument('--image-size', nargs='?', type=int, default=256,
                        help='训练时图像尺寸')
    parser.add_argument('--scale', nargs='?', type=bool,
                        default=False, const=True,
                        help='是否启用缩放数据增强（随机裁剪或缩放填充）')
    
    # 训练相关
    parser.add_argument('--n-epoch', nargs='?', type=int, default=1000,
                        help='训练轮数')
    parser.add_argument('--feature-scale', nargs='?', type=int, default=1,
                        help='特征缩放因子（用于调整模型特征数）')
    
    # 检查点相关
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='预训练模型权重文件路径 (.pkl)')
    parser.add_argument('--furukawa-weights', nargs='?', type=str, default=None,
                        help='Furukawa预训练模型权重文件路径 (.pkl)')
    parser.add_argument('--new-hyperparams', nargs='?', type=bool,
                        default=False, const=True,
                        help='是否使用新的超参数继续训练（不加载优化器状态）')
    
    # 日志和调试相关
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='日志目录路径')
    parser.add_argument('--debug', nargs='?', type=bool,
                        default=False, const=True,
                        help='是否启用调试模式（单进程，便于调试）')
    parser.add_argument('--plot-samples', nargs='?', type=bool,
                        default=False, const=True,
                        help='是否在TensorBoard中绘制示例预测图像')
    
    args = parser.parse_args()

    # ========== 初始化日志和TensorBoard ==========
    log_dir = args.log_path + '/' + time_stamp + '/'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)  # TensorBoard写入器
    
    # 配置日志记录器
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/train.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 开始训练
    train(args, log_dir, writer, logger)

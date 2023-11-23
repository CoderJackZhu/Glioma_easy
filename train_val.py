import gc
import glob
import os
import sys

import torch.backends.cudnn as cudnn
import torch.optim
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
# nnU-Net的图像增强方法
# from dataset.transform import GaussianNoise, GaussianBlur, BrightnessMultiplicative, \
#     ContrastAugmentation, SimulateLowResolution, Gamma, Mirror
from dataset.transform import *
# from dataset.transform import MedicalImageScaler
from models import uniformerv2_b16

FILE = Path(__file__).resolve()

ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import init_logger, AverageMeter
from utils import get_scheduler, parser

from dataset import ClsDataset, ClsDatasetH5py


def train(device, args):
    check_rootfolders()
    if args.start_epoch != 0:
        runs = sorted(glob.glob(os.path.join(args.output, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(args.output, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(args.output, 'run', 'run_' + str(run_id))

    logger = init_logger(log_file=args.output + f'/log.txt')
    # 使用tensorboard可视化，每个新的
    tb_writer = SummaryWriter(log_dir=save_dir)
    val_dataset = ClsDatasetH5py(list_file=args.val_list,
                                 h5py_path='/media/spgou/DATA/ZYJ/Dataset/UCSF_TCIA_ROI_images_h5py',
                                 transform=[Resize((128, 128, 128),
                                                   # orig_shape=(155, 240, 240)
                                                   ),
                                            # RandomAugmentation((16, 16, 16), (0.8, 1.2), (0.8, 1.2)),
                                            ],

                                 )

    # train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    # val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
    train_dataset = ClsDatasetH5py(list_file=args.train_list,
                                   h5py_path='/media/spgou/DATA/ZYJ/Dataset/UCSF_TCIA_ROI_images_h5py',
                                   transform=[
                                       RandomAugmentation((16, 16, 16), (0.8, 1.2), (0.8, 1.2)),
                                       # GaussianNoise()
                                   ]
                                   )
    # [RandomAugmentation((16, 16, 16), (0.8, 1.2), (0.8, 1.2), (0.8, 1.2)),
    #                    ToTensor()]
    logger.info(f"Num train examples = {len(train_dataset)}")
    logger.info(f"Num val examples = {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers, pin_memory=True,
                                               drop_last=True)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # model = ClsModel(args.model_name, args.num_classes, args.is_pretrained)
    # model = generate_model(model_depth=args.model_depth)
    # model = MultiModalCNN(num_modalities=2, input_channels=4, hidden_channels=64,  num_classes=4)
    model = uniformerv2_b16(
        in_channels=4,
        input_resolution=128,
        pretrained=False,
        t_size=128, backbone_drop_path_rate=0.2, drop_path_rate=0.4,
        dw_reduction=1.5,
        no_lmhra=True,
        temporal_downsample=False,
        num_classes=args.num_classes
    )
    # model = UNETR(
    #     in_channels=4,
    #     out_channels=2,
    #     img_size=(128, 128, 128),
    #     num_classes=4,
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     norm_name='instance',
    #     conv_block=True,
    #     res_block=True,
    #     dropout_rate=0.0,
    # )
    # model = generate_model(model_depth=args.model_depth, n_classes=args.num_classes)

    print(model.state_dict().keys())
    model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    cudnn.benchmark = True

    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')

    model.zero_grad()
    eval_results = []
    eval_accs = []
    # scaler = GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        train_preds, train_labels = [], []
        losses = AverageMeter()
        model.train()

        for step, (img, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # with torch.autocast(device_type="cuda"):
            # # 把图片由双精度变为浮点数
            img = img.type(torch.cuda.FloatTensor)
            img = img.to(device)
            target = target.view(-1).to(device)

            output = model(img)
            # 梯度缩放

            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # scaler.step(optimizer)
            # scaler.update()
            predict = torch.max(output, dim=1)[1]
            train_preds.append(predict)
            train_labels.append(target)

            loss = criterion(output, target.long())
            loss.backward()
            train_preds.append(output.argmax(1))
            train_labels.append(target)
            losses.update(loss.item(), img.size(0))

            if step % args.print_freq == 0:
                logger.info(
                    f"Epoch: [{epoch}/{args.epochs}][{step}/{len(train_loader)}], lr: {optimizer.param_groups[-1]['lr']:.8f} \t loss = {losses.val:.4f}({losses.avg:.4f})")

            optimizer.step()
        scheduler.step()
        # 得到一个epoch的平均loss，并可视化
        logger.info(f'Epoch: [{epoch}/{args.epochs}] \t loss = {losses.avg:.4f}')
        tb_writer.add_scalar('train/loss', losses.avg, epoch + 1)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                preds, labels, losses = [], [], []
                eval_pbar = tqdm.tqdm(val_loader, desc=f'epoch {epoch + 1} / {args.epochs} evaluating', position=1,
                                      disable=False)
                for step, (img, target) in enumerate(eval_pbar):
                    optimizer.zero_grad()  # 添加这一行以确保梯度不会累积

                    # with torch.autocast(device_type="cuda"):
                    # 把图片由双精度变为浮点数
                    img = img.type(torch.cuda.FloatTensor)
                    img = img.to(device)
                    target = target.view(-1).to(device)

                    output = model(img)
                    loss = criterion(output, target.long())
                    predict = torch.max(output, dim=1)[1]

                    labels.append(target)
                    preds.append(predict)
                    losses.append(loss.item())

                labels = torch.cat(labels, dim=0)
                predicts = torch.cat(preds, dim=0)

                labels = labels.cpu().numpy()
                predicts = predicts.cpu().numpy()

                # 计算验证集的损失
                eval_loss = np.mean(losses)
                print(labels, predicts, eval_loss)

                eval_result = (np.sum(labels == predicts)) / len(labels)
                eval_results.append(eval_result)

                # 计算验证集的accuracy
                eval_acc = accuracy_score(labels, predicts)
                eval_accs.append(eval_acc)

                logger.info(f'precision = {eval_result:.4f}')
                logger.info(f'eval_loss = {eval_loss:.4f}')
                logger.info(f'eval_acc = {eval_acc:.4f}')

                # tensorboard可视化
                tb_writer.add_scalar('val/precision', eval_result, epoch + 1)
                tb_writer.add_scalar('val/loss', eval_loss, epoch + 1)
                tb_writer.add_scalar('val/accuracy', eval_acc, epoch + 1)

                # 保存模型
                save_path = os.path.join(args.output, f'precision_{eval_result:.4f}_num_{epoch + 1}')
                os.makedirs(save_path, exist_ok=True)
                model_to_save = (model.module if hasattr(model, "module") else model)
                torch.save(model_to_save.state_dict(), os.path.join(save_path, f'epoch_{epoch + 1}.pth'))

        train_preds = torch.cat(train_preds, dim=0).cpu().numpy()
        train_labels = torch.cat(train_labels, dim=0).cpu().numpy()
        train_accuracy = accuracy_score(train_labels, train_preds)
        logger.info(f'train_accuracy = {train_accuracy:.4f}')
        tb_writer.add_scalar('train/accuracy', train_accuracy, epoch + 1)
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.output]
    for folder in folders_util:
        os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.train_list)

    print(args.val_list)
    print(f'device: {device}')

    train(device, args)

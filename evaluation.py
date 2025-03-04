import sys
from pathlib import Path

import torch.distributed as dist
import torch.nn.parallel
import torch.optim

from dataset.cls_dataset import ClsDataset, ClsDatasetH5py
from dataset.transform import Resize
from utils.util import plot_confusion_matrix
from utils.util import plot_roc

FILE = Path(__file__).resolve()

ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import init_logger, torch_distributed_zero_first, distributed_concat
from utils import parser

import os

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import uniformerv2_b16


def evaluate(local_rank, device, args):
    check_rootfolders()
    logger = init_logger(log_file=args.output + f'/log')

    with torch_distributed_zero_first(rank):
        val_dataset = ClsDatasetH5py(
            list_file=args.test_list,
            h5py_path='/media/spgou/FAST/UCSF/UCSF-PDGM-v3-20230111_ROI_images_h5py',
            mode='test',
        )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers, pin_memory=True)

    print('val_loader is ready!!!')

    # model = ClsModel(args.model_name, args.num_classes, args.is_pretrained)
    # model = generate_model(model_depth=args.model_depth)
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

    if args.tune_from and os.path.exists(args.tune_from):
        print(f'loading model from {args.tune_from}')
        sd = torch.load(args.tune_from, map_location='cpu')
        model.load_state_dict(sd)
    else:
        raise ValueError("the path of model weights is not exist!")

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)

    cudnn.benchmark = True

    for k, v in sorted(vars(args).items()):
        logger.info(f'{k} = {v}')

    model.eval()
    with torch.no_grad():
        preds, labels, scores = [], [], []
        eval_pbar = tqdm.tqdm(val_loader, desc=f'evaluating', position=1, disable=False if rank in [-1, 0] else True)
        for step, (img, target) in enumerate(eval_pbar):
            # 变为浮点数
            img = img.type(torch.cuda.FloatTensor)
            img = img.to(device)
            # target = target.view(1, -1).expand(img.size(0), -1)
            # 更改target维度匹配输出
            target = target.view(-1)
            target = target.to(device, dtype=torch.float32)

            output = model(img)

            score = torch.softmax(output, dim=1)
            predict = torch.max(output, dim=1)[1]
            labels.append(target)
            scores.append(score)
            preds.append(predict)
        labels = torch.cat(labels, dim=0)
        predicts = torch.cat(preds, dim=0)
        scores = torch.cat(scores, dim=0)
        if rank != -1:
            labels = distributed_concat(labels, len(val_dataset))
            predicts = distributed_concat(predicts, len(val_dataset))
            scores = distributed_concat(scores, len(val_dataset))
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        predicts = predicts.cpu().numpy()
        if rank == 0:
            from sklearn import metrics
            report = metrics.classification_report(labels, predicts,
                                                   target_names=['{}'.format(x) for x in range(args.num_classes)],
                                                   digits=4, labels=range(args.num_classes))

            confusion = metrics.confusion_matrix(labels, predicts)
            # ROC曲线
            fpr, tpr, thresholds = metrics.roc_curve(labels, scores[:, 1], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            accuracy = metrics.accuracy_score(labels, predicts)
            precision = metrics.precision_score(labels, predicts, average='macro')
            recall = metrics.recall_score(labels, predicts, average='macro')
            f1 = metrics.f1_score(labels, predicts, average='macro')
            sensitivity = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])
            specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
            print('sensitivity: %.4f' % sensitivity)
            print('specificity: %.4f' % specificity)
            print('accuracy: %.4f' % accuracy)
            print('precision: %.4f' % precision)
            print('recall: %.4f' % recall)
            print('f1-score: %.4f' % f1)
            print('auc: %.4f' % auc)
            print(report)
            performance = np.sum(labels == predicts) / len(labels)
            print(performance)
            np.save(os.path.join(args.output, f"labels"), labels)
            np.save(os.path.join(args.output, f"scores"), scores)
            np.save(os.path.join(args.output, f"predicts"), predicts)
            np.save(os.path.join(args.output, f"fpr"), fpr)
            np.save(os.path.join(args.output, f"tpr"), tpr)
            # 作图
            plot_roc(fpr, tpr, auc, args.output)
            plot_confusion_matrix(confusion, args.output)
            with open(os.path.join(args.output, f"report.txt"), 'w') as f:
                f.write(report)
            logger.info(args.output)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.output]
    for folder in folders_util:
        os.makedirs(folder, exist_ok=True)


def distributed_init(backend="gloo", port=None):
    num_gpus = torch.cuda.device_count()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


if __name__ == '__main__':
    args = parser.parse_args()
    # 设置用第1块卡
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["RANK"] = '0'
    os.environ["WORLD_SIZE"] = '1'
    os.environ["LOCAL_RANK"] = '0'
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    distributed_init(backend=args.backend)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    args.world_size = int(os.environ["WORLD_SIZE"])

    print(f"[init] == local rank: {local_rank}, global rank: {rank} == devices: {device}")

    evaluate(local_rank, device, args)

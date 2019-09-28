import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import random

import mxnet as mx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import tqdm

from dataset import MXDataset
from transforms import get_train_transform, get_test_transform
from plot import plot_all
from serialization import save_dict
from models import get_pnasnet5large
from metric import get_score, binarize_prediction


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    mx.random.seed(random_seed)
    
    
def make_folds(df, n_folds, random_seed):
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=random_seed).itertuples(),
                          leave=False,
                          desc=f'[ Making {n_folds} folds..]',
                          total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds

    
def epoch_step(loader, desc, fp16, ctx, net, loss, trainer=None):
    loc_loss = n = 0
    all_predictions, all_targets = [], []
    with tqdm.tqdm(loader,
                   desc=desc,
                   mininterval=2,
                   leave=False,
                  ) as pbar:
        for i, batch in enumerate(pbar):
            data, label = batch
            if fp16:
                data = data.astype('float16', copy=False)
            data = mx.gluon.utils.split_and_load(data, ctx, even_split=False)
            label = mx.gluon.utils.split_and_load(label, ctx, even_split=False)
            if i > 0:
                for l in losses:
                    loc_loss += l.sum().asscalar()

                all_predictions.extend([c.asnumpy() for c in preds_np])
                all_targets.extend([y.asnumpy() for y in label_np])
                
                postfix = {'loss': f'{loc_loss/n:.3}'}
                if trainer is not None:
                    postfix.update({'lr': f'{trainer.learning_rate:.3}'})
                pbar.set_postfix(**postfix)
                
            if trainer is not None:
                with mx.autograd.record():
                    preds = [net(X) for X in data]
                    if fp16:
                        losses = [loss(X.astype('float32', copy=False), Y)
                                  for X, Y in zip(preds, label)]
                    else:
                        losses = [loss(X, Y)
                                  for X, Y in zip(preds, label)]
                for l in losses:
                    l.backward()

                trainer.step(batch[0].shape[0])
            else:
                preds = [net(X) for X in data]
                if fp16:
                    losses = [loss(X.astype('float32', copy=False), Y) for X, Y in zip(preds, label)]
                else:
                    losses = [loss(X, Y) for X, Y in zip(preds, label)]
                
            preds_np = [P.astype('float32', copy=False).sigmoid() for P in preds]
            label_np = label

            n += batch[0].shape[0]

        for l in losses:
            loc_loss += l.sum().asscalar()

        all_predictions.extend([c.asnumpy() for c in preds_np])
        all_targets.extend([y.asnumpy() for y in label_np])

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    return loc_loss/n, all_predictions, all_targets    


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data_path', type=str, default='data')
    
    arg('--model',    type=str, default='pnasnet5large')
    arg('--exp-name', type=str, default='pnasnet5large_2')
    
    arg('--batch-size', type=int, default=32)
    arg('--lr',         type=float, default=1e-2)
    arg('--patience',   type=int, default=4)
    arg('--n-epochs',   type=int, default=15)
    
    arg('--n-folds', type=int, default=10)
    arg('--fold',    type=int, default=0)
    
    arg('--random-seed', type=int, default=314159)
    
    arg('--num-workers', type=int, default=6)
    arg('--gpus', type=str, default='0')
    
    arg('--resize', type=int, default=331)
    arg('--crop',   type=int, default=331)
    arg('--scale',  type=str, default='0.4, 1.0')
    arg('--mean',   type=str, default='0.485, 0.456, 0.406')
    arg('--std',    type=str, default='0.229, 0.224, 0.225')
    
    args = parser.parse_args()
    print(args)
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    #  os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
    #  os.environ['MXNET_UPDATE_ON_KVSTORE'] = "0"
    #  os.environ['MXNET_EXEC_ENABLE_ADDTO'] = "1"
    #  os.environ['MXNET_USE_TENSORRT'] = "0"
    #  os.environ['MXNET_GPU_WORKER_NTHREADS'] = "2"
    #  os.environ['MXNET_GPU_COPY_NTHREADS'] = "1"
    #  os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = "54"
    
    random_seed = args.random_seed
    set_random_seed(random_seed)
    
    path_to_data = Path(args.data_path)
    labels = pd.read_csv(path_to_data / 'labels.csv')
    num_classes = len(labels)
    
    train = pd.read_csv(path_to_data / 'train.csv.zip')
    
    n_folds = args.n_folds
    make_folds(train, n_folds, random_seed)
    
    mlb = MultiLabelBinarizer([str(i) for i in range(num_classes)])
    s = train['attribute_ids'].str.split()
    res = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=train.index)
    train = pd.concat([res, train['id'] + '.png', train['fold']], axis=1)
    
    gpu_count = len(args.gpus.split(','))
    batch_size = args.batch_size
    
    resize = args.resize
    crop = args.crop
    scale = tuple(float(x) for x in args.scale.split(','))
    mean = [float(x) for x in args.mean.split(',')]
    std = [float(x) for x in args.std.split(',')]

    #  jitter_param = 0.4
    #  lighting_param = 0.1
    labels_ids = [str(i) for i in range(num_classes)]
    num_workers = args.num_workers
    
    fold = args.fold
    train_transformer = get_train_transform(resize=resize,
                                            crop=crop,
                                            scale=scale,
                                            mean=mean,
                                            std=std)
    train_loader = mx.gluon.data.DataLoader(MXDataset(path_to_data / 'train',
                                                      train[train['fold'] != fold].copy(),
                                                      labels_ids,
                                                      train_transformer),
                                            batch_size=batch_size*gpu_count,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)

    test_transformer = get_test_transform(resize=resize,
                                          crop=crop,
                                          mean=mean,
                                          std=std)
    dev_loader = mx.gluon.data.DataLoader(MXDataset(path_to_data / 'train',
                                                    train[train['fold'] == fold].copy(),
                                                    labels_ids,
                                                    test_transformer),
                                          batch_size=batch_size*gpu_count,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=True)
    fp16 = True
    if args.model == 'pnasnet5large':
        net = get_pnasnet5large(num_classes)
    else:
        raise(f'No such model {args.model}')
    
    if fp16:
        net.cast('float16')
    ctx = [mx.gpu(i) for i in range(gpu_count)]
    net.collect_params().reset_ctx(ctx)
    
    epoch_size = len(train_loader)
    lr = args.lr*batch_size/256
    steps = [step*epoch_size for step in [7, 9]]
    factor = 0.5
    warmup_epochs = 5
    warmup_mode = 'linear'
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps,
                                                    factor=factor,
                                                    base_lr=lr,
                                                    warmup_steps=warmup_epochs*epoch_size,
                                                    warmup_mode=warmup_mode)
    
    if fp16:
        weight = 128
        opt = mx.optimizer.Adam(multi_precision=True, learning_rate=lr, rescale_grad=1/weight,
                                lr_scheduler=schedule,
                               )
    else:
        opt = mx.optimizer.Adam(learning_rate=lr,
                                lr_scheduler=schedule,
                               )
    trainer = mx.gluon.Trainer(net.collect_params(), opt)
    if fp16:
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(weight=weight)
    else:
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    
    path_to_models = Path('models')
    path_to_model = path_to_models / args.exp_name
    path_to_exp = path_to_model / f'fold_{fold}'
    if not path_to_exp.exists():
        path_to_exp.mkdir(parents=True)
        
    patience = args.patience
    lr_reset_epoch = 1
    lr_changes = 0
    max_lr_changes = 2
    n_epochs = args.n_epochs
    best_dev_f2 = th2 = 0
    train_losses = []
    dev_losses, dev_f2s, dev_ths = [], [], []
    dev_met1, dev_met2 = [], []
    for epoch in range(1, n_epochs + 1):
        train_loss, all_predictions, all_targets = epoch_step(train_loader,
                                                              desc=f'[ Training {epoch}/{n_epochs}.. ]',
                                                              fp16=fp16,
                                                              ctx=ctx, net=net, loss=loss, trainer=trainer)
        train_losses.append(train_loss)

        dev_loss, all_predictions, all_targets = epoch_step(dev_loader,
                                                            desc=f'[ Validating {epoch}/{n_epochs}.. ]',
                                                            fp16=fp16,
                                                            ctx=ctx, net=net, loss=loss)
        dev_losses.append(dev_loss)

        metrics = {}
        argsorted = all_predictions.argsort(axis=1)
        for threshold in [0.01, 0.05, 0.1, 0.15, 0.2]:
            metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
                binarize_prediction(all_predictions, threshold, argsorted), all_targets)
        dev_met1.append(metrics)
        
        dev_f2 = 0
        for th in dev_met1[-1]:
            if dev_met1[-1][th] > dev_f2:
                dev_f2 = dev_met1[-1][th]
                th2 = th

        all_predictions = all_predictions/all_predictions.max(1, keepdims=True)
        metrics = {}
        argsorted = all_predictions.argsort(axis=1)
        for threshold in [0.05, 0.1, 0.2, 0.3, 0.4]:
            metrics[f'valid_norm_f2_th_{threshold:.2f}'] = get_score(
                binarize_prediction(all_predictions, threshold, argsorted), all_targets)
        dev_met2.append(metrics)

        for th in dev_met2[-1]:
            if dev_met2[-1][th] > dev_f2:
                dev_f2 = dev_met2[-1][th]
                th2 = th

        dev_f2s.append(dev_f2)
        dev_ths.append(th2)
        if dev_f2 > best_dev_f2:
            best_dev_f2 = dev_f2
            best_th = th2
            if fp16:
                net.cast('float32')
                net.save_parameters((path_to_exp / 'model').as_posix())
                net.cast('float16')
            else:
                net.save_parameters((path_to_exp / 'model').as_posix())
            save_dict(
                {
                    'dev_loss': dev_loss,
                    'dev_f2': best_dev_f2,
                    'dev_th': best_th,
                    'epoch': epoch,
                    'dev_f2s': dev_f2s,
                    'dev_ths': dev_ths,

                    'dev_losses': dev_losses,
                    'dev_met1': dev_met1,
                    'dev_met2': dev_met2,
                },
                path_to_exp / 'meta_data.pkl')
        elif (patience and epoch - lr_reset_epoch > patience and
              max(dev_f2s[-patience:]) < best_dev_f2):
            # "patience" epochs without improvement
            lr_changes +=1
            if lr_changes > max_lr_changes:
                break
            lr *= factor
            print(f'lr updated to {lr}')
            lr_reset_epoch = epoch
            if fp16:
                weight = 128
                opt = mx.optimizer.Adam(multi_precision=True, learning_rate=lr, rescale_grad=1/weight)
            else:
                opt = mx.optimizer.Adam(learning_rate=lr)
            trainer = mx.gluon.Trainer(net.collect_params(), opt)

        plot_all(path_to_exp, train_losses, dev_losses, dev_f2s, dev_ths, dev_met1, dev_met2)


if __name__ == '__main__':
    main()

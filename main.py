# code from # and https://github.com/ylsung/pytorch-adversarial-training/blob/master/cifar-10/main.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

import torchvision as tv
import random
import numpy as np
from time import time
from madry_model import WideResNet
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from argument import parser, print_args

os.chdir("/home/projects/DATNS")


class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_dataset, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.SGD(model.parameters(), args.learning_rate, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[40000, 60000],
                                                         gamma=0.1)

        counter = 0
        class_epoch_counter = 0
        _iter = 0

        s_0 = 100
        sp = 2
        # class_sp = 4

        #sp = random.sample(range(s_0, 200), 63)


        begin_time = time()
        for epoch in range(1, args.max_epoch+1):

            if epoch >= s_0:
                if (epoch % sp) == 0:
                    counter += 1
                    adv_train = True
                    print("Switching to Adversarial Training at epoch {}".format(epoch))
                else:
                    adv_train = False

            tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=4)
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    with ctx_noparamgrad_and_eval(model):
                        adv_data = self.attack.perturb(data, label)
                    output = model(adv_data)
                else:
                    output = model(data)

                loss = F.cross_entropy(output, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, f'checkpoint_{_iter}.pth')
                    save_model(model, file_name)

                _iter += 1
                scheduler.step()

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc, nat_cm, adv_cm = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n'+'='*20 +f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                    +'='*20)
                logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, spent: {t2-t1:.3f} s')
                logger.info('='*28+' end of evaluation '+'='*28+'\n')



    def test(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        nb_classes = 9

        # Initialize the prediction and label lists(tensors)
        predlist_natural = torch.zeros(0, dtype=torch.long, device='cpu')
        lbllist_natural = torch.zeros(0, dtype=torch.long, device='cpu')

        predlist_adv = torch.zeros(0, dtype=torch.long, device='cpu')
        lbllist_adv = torch.zeros(0, dtype=torch.long, device='cpu')

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, label)

                    adv_output = model(adv_data)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

                # here we need to gather all the predictions on natural and adversarial samples
                # Append batch prediction results
                predlist_natural = torch.cat([predlist_natural, pred.view(-1).cpu()])
                lbllist_natural = torch.cat([lbllist_natural, label.view(-1).cpu()])

                predlist_adv = torch.cat([predlist_adv, adv_pred.view(-1).cpu()])
                lbllist_adv = torch.cat([lbllist_adv, label.view(-1).cpu()])

            # Confusion matrices
            conf_mat_natural = confusion_matrix(lbllist_natural.numpy(), predlist_natural.numpy())
            #print(conf_mat_natural)

            conf_mat_adv = confusion_matrix(lbllist_adv.numpy(), predlist_adv.numpy())
            #print(conf_mat_adv)

            # # Per-class accuracy
            # class_accuracy_natural = 100 * conf_mat_natural.diagonal() / conf_mat_natural.sum(1)
            # print(class_accuracy_natural)
            #
            # class_accuracy_adv = 100 * conf_mat_adv.diagonal() / conf_mat_adv.sum(1)
            # print(class_accuracy_adv)
            #
            # classes = ('plane', 'car', 'bird', 'cat', 'deer',
            #            'dog', 'frog', 'horse', 'ship', 'truck')
            #
            # plt.plot(class_accuracy_natural, label='natural')
            # plt.plot(class_accuracy_adv, label='adversarial')
            # plt.xlabel("classes")
            # plt.ylabel("Accuracy")
            # plt.legend()
            # plt.xticks(ticks=range(0, len(classes)), labels=classes, rotation=45)
            # plt.savefig("per_class_accuracies.png")
            # plt.show()
            # plt.close()
            #
            # plt.figure(figsize=(15, 10))
            #
            # #class_names = list(label2class.values())
            # df_cm = pd.DataFrame(conf_mat_natural, index=classes, columns=classes).astype(int)
            # heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
            #
            # heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
            # heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
            # plt.ylabel('True label')
            # plt.xlabel('Predicted label')
            # #plt.figure(figsize=(15, 10))
            #
            # plt.savefig("cm_natural.png")
            # plt.show()
            # # plt.draw()
            # plt.close()
            #
            # plt.figure(figsize=(15, 10))
            #
            # # class_names = list(label2class.values())
            # df_cm = pd.DataFrame(conf_mat_adv, index=classes, columns=classes).astype(int)
            # heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
            #
            # heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
            # heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
            # plt.ylabel('True label')
            # plt.xlabel('Predicted label')
            # #plt.figure(figsize=(15, 10))
            #
            # plt.savefig("cm_adv.png")
            # plt.show()
            # # plt.draw()
            # plt.close()
            #
            # print('Classification Report (Natural Accuracy)')
            # print(classification_report(lbllist_natural, predlist_natural))
            # print('Classification Report (Adversarial Accuracy)')
            # print(classification_report(lbllist_adv, predlist_adv))

        return total_acc / num, total_adv_acc / num


def main(args):

    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    #model = WideResNet(depth=16, num_classes=10, widen_factor=10, dropRate=0.0)
    generator = tv.models.resnet18(pretrained=True)
    print(generator)

    attack = LinfPGDAttack(generator,
                           loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                           eps=8 / 255.0,
                           nb_iter=10,
                            eps_iter=2 / 255.0,
                           rand_init=True,
                           clip_min=0.0,
                           clip_max=1.0,
                           targeted=False)


    if torch.cuda.is_available():
        generator.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding=4),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])


        tr_dataset = tv.datasets.CIFAR10(args.data_root,
                                              train=True,
                                              transform=transform_train,
                                              download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR10(args.data_root,
                                       train=False,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


        total = time()
        trainer.train(generator, tr_dataset, tr_loader, te_loader)
        import time as t
        t.sleep(5)
        print('{} seconds'.format(time() - total))

    elif args.todo == 'test':
        te_dataset = tv.datasets.CIFAR10(args.data_root,
                                       train=False,
                                       transform=tv.transforms.ToTensor(),
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        checkpoint = torch.load(args.load_checkpoint)
        generator.load_state_dict(checkpoint)

        std_acc, adv_acc = trainer.test(generator, te_loader, adv_test=True, use_pseudo_label=False)

        print(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")

    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)

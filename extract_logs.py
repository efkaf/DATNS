def extract_logs(dir, train=True):
    standard_tr_acc_list = []
    robustness_tr_acc_list = []

    with open(dir) as openfile:
        n_epochs = 0
        epochs = []
        count_iters = []
        epoch_number = -1
        epoch_counter = 0
        iters = []
        e = []
        if train:
            for line in openfile:
                if "epoch" in line and not "evaluation" in line:
                    epoch_number = line.split()[1]
                    epochs.append(epoch_number)
                    # if epoch_number not in epochs:
                    #     epochs.append(epoch_number)

                if "standard" in line:
                    #while epoch_number:
                    standard_tr_acc_list.append((line.split()[2]))
                    robustness_tr_acc_list.append((line.split()[-1]))
        else:
            for line in openfile:

                if "evaluation" in line and not "end" in line:
                    epoch_number = line.split()[-4]
                    print(epoch_number)
                    epochs.append(epoch_number)
                    #epochs.append(epoch_number)


                if "test" in line:
                    standard_tr_acc_list.append((line.split()[2]))
                    robustness_tr_acc_list.append((line.split()[-4]))


        for i in epochs:
            count_iters.append(epochs.count(i))

        for i in range(len(epochs)):
            if epochs[i] not in e:
                e.append(epochs[i])
                iters.append(count_iters[i])
        #print(iters)


        import numpy as np

        standard_tr_acc_list = np.array([i.strip('%,') for i in standard_tr_acc_list]).astype('float32')
        robustness_tr_acc_list = np.array([i.strip('%,') for i in robustness_tr_acc_list]).astype('float32')


        standard_tr_acc_per_epoch = []
        robustness_tr_acc_per_epoch = []

        start = 0
        for i in range(len(iters)-1):
            standard_tr_acc_per_epoch.append(sum(standard_tr_acc_list[start:start+iters[i]]) / iters[i])
            robustness_tr_acc_per_epoch.append(sum(robustness_tr_acc_list[start:start+iters[i]]) / iters[i])

            if (start < len(standard_tr_acc_list) - iters[i]):
                start = start + iters[i]
                #print(start)




        return standard_tr_acc_per_epoch, robustness_tr_acc_per_epoch



    # print(standard_tr_acc_per_epoch)
    # print(robustness_tr_acc_per_epoch)
    #



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # wrn 28x10 cifar10 from 100
    #proposed100 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/logs_paper/train_logs_wrn-20x10-cifar10-alternating_from_100.txt'
    #proposed75 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-10-wrn-from-75/train_log.txt'
    #proposed55 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-10-wrn-from-55/train_log.txt'


    # #gupta = '/home/efkaf/PycharmProjects/efficient-adversarial-training/gupta/log/cifar-10_wrn/train_log.txt'
    # madry = '/home/efkaf/PycharmProjects/efficient-adversarial-training/madry/log/cifar-10_wrn/train_log.txt'
    # gupta = '/home/efkaf/PycharmProjects/efficient-adversarial-training/gupta/log/cifar-10-wrn-from-100/train_log.txt'

    # rn18 cifar10
    # proposed100 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn18-from-55-random-73-rn18-pretrained-100-200-sp1/train_log.txt'
    # proposed75 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-10-rn18-from-75/train_log.txt'
    # proposed55 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn18-from-55-random-73-rn18-pretrained-55-200-sp1/train_log.txt'

    # rn50 cifar10
    #proposed100 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-10-rn50-from-100/train_log.txt'
    #proposed75 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-10-rn50-from-75/train_log.txt'
    #proposed55 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-10-rn50-from-55/train_log.txt'


    # rn50 cifar100
    # proposed100 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/logs_paper/rn50-cifar-100-from-100.txt'
    # proposed75 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn50-from-75/train_log.txt'
    # proposed55 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn50-from-55/train_log.txt'

    # rn18 cifar100
    # proposed100 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn18-from-100/train_log.txt'
    # proposed75 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn18-from-75/train_log.txt'
    # proposed55 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn18-from-55/train_log.txt'

    # plot f at 55
    # cifar100-rn18 f=2,3,5
    f2 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn18-from-55/train_log.txt'
    f3 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn-18-from-55-sp-3/train_log.txt'
    f5 = '/home/efkaf/PycharmProjects/efficient-adversarial-training/alt-switches/log/cifar-100-rn18-from-55-sp-5/train_log.txt'

    # cifar100-rn50 f=2,3,5


    # sa100,aa100 = extract_logs(proposed100, train=True)
    # sa75,aa75 = extract_logs(proposed75, train=True)
    # sa55,aa55 = extract_logs(proposed55, train=True)


    # plt.plot(sa55, label='natural/s0=55')
    # plt.plot(sa75, label='natural/s0=75')
    # plt.plot(sa100,label='natural/s0=100')
    #
    # plt.plot(aa55, label='adversarial/s0=55')
    # plt.plot(aa75, label='adversarial/s0=75')
    # plt.plot(aa100, "c",  label='adversarial/s0=100')

    sa2, aa2 = extract_logs(f2, train=True)
    sa3, aa3 = extract_logs(f3, train=True)
    sa5, aa5 = extract_logs(f5, train=True)

    plt.plot(sa2, label='natural/f=2')
    plt.plot(sa3, label = 'natural/f=3')
    plt.plot(sa5, label='natural/f=5')
    plt.plot(aa2, label='adversarial/f=2')
    plt.plot(aa3, label='adversarial/f=3')
    plt.plot(aa5, "c", label='adversarial/f=5')




    # plt.plot(aaG)
    plt.title('ResNet18 - CIFAR-100')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc="lower right")

    plt.show()
from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import numpy as np


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    conf_mat_final = np.zeros((dataset.dataset.nclasses, dataset.dataset.nclasses))
    all_labels = np.array([v for v in dataset.dataset.class_to_idx.values()])
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, conf_mat = model.test(all_labels)
        conf_mat_final += conf_mat
        writer.update_counter(ncorrect, nexamples)
    conf_mat_final = conf_mat_final / conf_mat_final.sum(1) * 100
    writer.print_acc(epoch, writer.acc)

    for l in all_labels:
        print('label %d, predictions: %s' % (l, conf_mat_final[l]))

    return writer.acc


if __name__ == '__main__':
    run_test()

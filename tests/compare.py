import argparse
import itertools
import torch
from detectron2.utils.file_io import loadTensors


def print_stat(x: torch.Tensor, name: str):
    print('{}.max(): {}, {}.min(): {}'.format(name, round(x.max().item(), 4), name, round(x.min().item()), 4))
    print('{}.mean: {}, {}.stddev: {}'.format(name, round(x.mean().item(), 4), name, round(x.std().item(), 4)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tensor1')
    parser.add_argument('tensor2')

    args = parser.parse_args()

    t1 = loadTensors(args.tensor1)
    t2 = loadTensors(args.tensor2)

    print('=' * 50)
    for idx, x1, x2 in zip(itertools.count(), t1, t2):
        x1 = x1.cuda().float()
        x2 = x2.cuda().float()
        if x1.shape[0] != 1:
            x1 = x1.unsqueeze(0)
        if x2.shape[0] != 1:
            x2 = x2.unsqueeze(0)
        if x1.shape == x2.shape:
            l1 = torch.norm(x1, 2)
            l2 = torch.norm(x2, 2)
            ld = torch.norm(x1 - x2, 2)
            delta = (x1 - x2).abs()
            max_diff = delta.max()
            mean_diff = delta.mean()
            std_diff = delta.std()
            cosine_sim = torch.mean(torch.nn.functional.cosine_similarity(x1.flatten(), x2.flatten(), dim=0))
            print('tensor{}: shape={}'.format(idx, x1.shape))
            print('cosine similarity: {}'.format(round(cosine_sim.item(), 4)))
            print('|x1| = {}, |x2| = {}, |x1 - x2| = {}'.format(round(l1.item(), 4), round(l2.item(), 4), round(ld.item(), 4)))
            print_stat(x1, 'x1')
            print_stat(x2, 'x2')
            print_stat(delta, '|x1 - x2|')
        else:
            print('Shape mismatch: {} != {}'.format(x1.shape, x2.shape))
        print('=' * 50)

import argparse
import itertools
import torch

def load_tensor(path: str):
    try:
        tensors = torch.load(path)
    except:
        tensors = torch.jit.load(path)._parameters.values()
    return [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tensor1')
    parser.add_argument('tensor2')

    args = parser.parse_args()

    t1 = load_tensor(args.tensor1)
    t2 = load_tensor(args.tensor2)

    print('=' * 50)
    for idx, x1, x2 in zip(itertools.count(), t1, t2):
        x1 = x1.cuda()
        x2 = x2.cuda()
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
            print('|x1| = {}, |x2| = {}, |x1 - x2| = {}'.format(round(l1.item(), 4), round(l2.item(), 4), round(ld.item(), 4)))
            print('x1.max(): {}, x1.min(): {}'.format(x1.max().item(), x1.min().item()))
            print('x2.max(): {}, x2.min(): {}'.format(x2.max().item(), x2.min().item()))
            print('cosine similarity: {}'.format(round(cosine_sim.item(), 4)))
            print('x1.mean: {}, x1.stddev: {}'.format(x1.mean().item(), x1.std().item()))
            print('x2.mean: {}, x2.stddev: {}'.format(x2.mean().item(), x2.std().item()))
            print('max difference: ', max_diff.item())
            print('mean difference: ', mean_diff.item())
            print('stddev difference: ', std_diff.item())
        else:
            print('Shape mismatch: {} != {}'.format(x1.shape, x2.shape))
        print('=' * 50)

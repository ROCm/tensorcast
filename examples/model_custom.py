import torch
import torchvision
import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
import tcast
from tqdm import tqdm
import time
import timm

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def imagenet_loader(args): 
    batch_size = args.batch_size
    num_workers = args.num_worker
    data_dir = args.data_dir 
    
    # see https://pytorch.org/vision/stable/models.html for setting transform
    transform = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(256), 
                        torchvision.transforms.CenterCrop(224),  
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                        ])
        
    train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'ILSVRC2012_img_train'),
                                                transform=transform)
        
    if not os.path.isfile(os.path.join(data_dir, 'wnid_to_label.pickle')):
        with open(os.path.join(data_dir, 'wnid_to_label.pickle'), 'wb') as f:
            pickle.dump(train_ds.class_to_idx, f)         

    test_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'ILSVRC2012_img_val'),
                                                transform=transform)
    g = torch.Generator()
    g.manual_seed(0)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)
    test_dl = DataLoader(test_ds, min(batch_size, 1024), shuffle=False,
                            num_workers=num_workers) 

    return train_dl, test_dl

def test_accuracy(model, test_dl, device, topk=(1, )):
    """ 
    Compute top k accuracy on testing dataset
    """
    start = time.time()
    model.to(args.device)
    model.eval()
    maxk = max(topk)
    topk_count = np.zeros((len(topk), len(test_dl)))
    
    for j, (x_test, target) in enumerate(tqdm(test_dl, "Evaluation")):
        with torch.no_grad():
            y_pred = model(x_test.to(device))
        topk_pred = torch.topk(y_pred, maxk, dim=1).indices
        target = target.to(device).view(-1, 1).expand_as(topk_pred)
        correct_mat = (target == topk_pred)

        for i, k in enumerate(topk):
            topk_count[i, j] = correct_mat[:, :k].reshape(-1).sum().item()

    topk_accuracy = topk_count.sum(axis=1) / len(test_dl.dataset)
    model.cpu()
    end = time.time()
    print(f'Time taken for inference on {args.model} model is {end - start} seconds.')
    print(f'Top-1 accuracy for {args.model} model is {topk_accuracy[0]}.')
    print(f'Top-5 accuracy for {args.model} model is {topk_accuracy[1]}.')

def get_model(args):
    if hasattr(torchvision.models, args.model):
        return getattr(torchvision.models, args.model)(pretrained=True)
    elif args.model in timm.list_models():
        return timm.create_model(args.model, pretrained=True)
    else:
        raise ValueError(f"Model {args.model} not found in torchvision or timm.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='torchvision model to load; pass `resnet50`, `mobilenet_v3_large`, or `inception_v4`')
    parser.add_argument(
        '--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument(
        '--data-dir', type=str, help='imagenet directory')
    parser.add_argument(
        '--batch-size', default=64, type=int, help='eval batch size')
    parser.add_argument(
        '--num-worker', default=2, type=int, help='number of workers for loading dataset')

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')
    set_seed(args.seed)

    train_loader, test_loader = imagenet_loader(args)
    
    model = get_model(args)
    test_accuracy(model, test_loader, args.device, (1, 5))

    # Using the custom layers in the model
    bfp16ebs8_t = tcast.DataType("int8", "e8m0_t8", "bfp16ebs8_t")
    tcast_specs = {'weight_dtype': bfp16ebs8_t, 'input_dtype': bfp16ebs8_t, 'output_dtype': bfp16ebs8_t}
    tcast.TorchInjector(tcast_specs)

    model_custom = get_model(args)
    test_accuracy(model_custom, test_loader, args.device, (1, 5))
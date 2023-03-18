import argparse
import os
import re
from typing import Dict
import numpy as np
import json
import random
import torchvision.transforms as T

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import PIL

from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIMSampler
from DiffusionFreeGuidence.ModelCondition import UNet
from SceneGraph.vis import draw_layout
from Scheduler import GradualWarmupScheduler
from data import imagenet_deprocess_batch
from data.vg import VgSceneGraphDataset, vg_collate_fn, vg_uncollate_fn

from utils import int_tuple, str_tuple
from utils import bool_flag

from torchsummary import summary
from data.utils import imagenet_preprocess, Resize, imagenet_deprocess,rescale

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank:int, world_size:int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
torch.cuda.empty_cache()
VG_DIR = os.path.expanduser('/media/kiki/971339f7-b775-448b-b7d8-f17bc1499e4d/kiki/Documents/Dataset/VG')
COCO_DIR = os.path.expanduser('datasets/coco')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])


# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64, 64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

# Optimization hyperparameters
parser.add_argument('--batch_size', default=20, type=int)
args = parser.parse_args()

modelConfig = {
        "state": "train",  # or eval
        "epoch": 200,
        "batch_size": 100,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition_only_obj_32/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_199_.pt",
        "noise_dir": "./NoiseImgs/",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs",
        "sampledImgName": "SampledGuidenceImgs_sheepbysheep_testddim",
        "test_fake_dir": "./TestFakeImgs/",
        "test_real_dir": "./TestRealImgs/",
        "test_layout_box_dir": "./TestLayoutBox/",
        "test_layout_img_dir": "./TestLayoutImg/",
        "test_sg_dir": "./TestSG/",
        "nrow": 1,
        "world_size": torch.cuda.device_count(),
    }

def build_vg_dsets(args):
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    dset_kwargs = {
            'vocab': vocab,
            'h5_path': args.train_h5,
            'image_dir': args.vg_image_dir,
            'image_size': args.image_size,
            'max_samples': args.num_train_samples,
            'max_objects': args.max_objects_per_image,
            'use_orphaned_objects': args.vg_use_orphaned_objects,
            'include_relationships': args.include_relationships,
    }
    train_dset = VgSceneGraphDataset(**dset_kwargs)
    iter_per_epoch = len(train_dset) // args.batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    dset_kwargs['h5_path'] = args.val_h5
    del dset_kwargs['max_samples']
    val_dset = VgSceneGraphDataset(**dset_kwargs)

    return vocab, train_dset, val_dset


def build_loaders(args):
    vocab, train_dset, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn

    train_loader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': args.loader_num_workers,
            'shuffle': False,
            'collate_fn': collate_fn,
            'drop_last': True,
    }

    val_loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': False,
        'collate_fn': collate_fn,
    }

    train_loader = DataLoader(train_dset, **train_loader_kwargs, sampler=DistributedSampler(train_dset))
    val_loader = DataLoader(val_dset, **val_loader_kwargs)
    return vocab, train_loader, val_loader


def build_val_loaders(args):
    vocab, _, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn

    val_loader_kwargs = {
        # 'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': False,
        'collate_fn': collate_fn,
    }

    val_loader = DataLoader(val_dset, **val_loader_kwargs)
    return vocab, val_loader


def train(rank:int, world_size):

    ddp_setup(rank, world_size)
    gpu_id = rank

    # VG dataset
    vocab, train_loader, val_loader = build_loaders(args)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"], vocab=vocab)
    net_model = net_model.to(gpu_id)
    net_model = DDP(net_model, device_ids=[gpu_id], find_unused_parameters=True)

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"])), strict=False)
        print("Model weight load down.")

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 1, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])# .to(device)
    trainer = trainer.to(gpu_id)
    trainer = DDP(trainer, device_ids=[gpu_id],find_unused_parameters=True)

    print(sum(p.numel() for p in net_model.parameters() if p.requires_grad))

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                if modelConfig["training_load_weight"] is not None:
                    org_ep = re.findall(r'\d+', modelConfig["training_load_weight"])[0]
                    ep_num = e+int(org_ep)+1
                else:
                    ep_num = e
                images, objs, boxes, triples, obj_to_img, triple_to_img = batch
                # labels = [objs.cuda(), boxes.cuda(), triples.cuda(), obj_to_img.cuda(), triple_to_img.cuda()]
                labels = [objs.to(gpu_id), boxes.to(gpu_id), triples.to(gpu_id), obj_to_img.to(gpu_id), triple_to_img.to(gpu_id)]

                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(gpu_id)
                # x_0 = images.cuda()
                loss = trainer(x_0, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        if gpu_id == 0:
            torch.save(net_model.module.state_dict(), os.path.join(
                modelConfig["save_dir"], 'ckpt_' + str(ep_num) + "_.pt"))
    destroy_process_group()

def sample(modelConfig: Dict):
    scene_graph =    {
    "objects": ["sky", "grass", "sheep", "sheep", "tree", "ocean", "boat"],
    "relationships": [
      [0, "above", 1],
      [2, "standing on", 1],
      [3, "by", 2],
      [4, "behind", 2],
      [5, "by", 4],
      [6, "on", 1]
       ]
    }

    vocab, val_loader = build_val_loaders(args)
    device = torch.device(modelConfig["device"])
    # objs, triples, obj_to_img = encode_scene_graphs(scene_graph, vocab, device)
    # labels = [objs.to(device), None, triples.to(device), obj_to_img.to(device), None]
    save_sg_path = modelConfig["test_sg_dir"]
    save_layout_box_path = modelConfig["test_layout_box_dir"]
    save_layout_img_path = modelConfig["test_layout_img_dir"]
    sampled_dir = modelConfig["test_fake_dir"]
    real_dir = modelConfig["test_real_dir"]
    device = torch.device(modelConfig["device"])

    # load model and evaluate
    with torch.no_grad():

        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"], vocab=vocab).to(device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        # DDIM
        sampler = DDIMSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)

        # # DDPM
        # sampler = GaussianDiffusionSampler(
        #     model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)

        image_size = (32, 32)
        all_imgs = []
        deprocess_fn = imagenet_deprocess(rescale_image=rescale)
        transforms = [Resize(image_size), T.ToTensor()]
        transform = T.Compose(transforms)

        ep = 0
        b = 1
        it = 0

        with tqdm(val_loader, dynamic_ncols=True) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                ep += 1
                images, objs, boxes, triples, obj_to_img, triple_to_img = batch

                ''''' save the layout boxes '''''
                # draw_layout(vocab, objs, boxes, save_layout_box_path, str(b)+'_'+str(it)+'.png', masks=None, size=32,
                #             show_boxes=True, bgcolor='white', bg=[])

                """ save the scene graphs txt file"""
                # out = vg_uncollate_fn(batch)
                # for i in range(len(out)):
                #     out_i = out[i]
                #     obj_i = out_i[1]
                #     tuple_i = out_i[3]
                #     tup_num = len(tuple_i)
                #     with open(save_sg_path+str(ep)+'_'+str(i)+'.txt', 'w') as f:
                #         for j in range(tup_num):
                #             t = tuple_i[j]
                #             o_i, p_i, s_i = obj_i[t[0]], t[1], obj_i[t[2]]
                #             o, p, s = vocab['object_idx_to_name'][o_i], vocab['pred_idx_to_name'][p_i], vocab['object_idx_to_name'][s_i]
                #             f.write(o+','+p+','+s+'\n')

                """  generate fake image"""
                labels = [objs.to(device), boxes.to(device), triples.to(device), obj_to_img.to(device),
                          triple_to_img.to(device)]
                # Sampled from standard normal distribution
                noisyImage = torch.randn(
                    size=[100, 3, modelConfig["img_size"], modelConfig["img_size"]],
                    device=device)

                # generate
                sampledImgs = sampler(noisyImage, labels)
                sampledImgs = imagenet_deprocess_batch(sampledImgs)

                for i in range(sampledImgs.size(0)):
                    save_image(sampledImgs[i, :, :, :], os.path.join(
                        modelConfig["test_fake_dir"], str(ep) + '_{}.png'.format(i)))

                """ save the real images"""
                # images = imagenet_deprocess_batch(images)
                # for i in range(images.size(0)):
                #     save_image(images[i, :, :, :], os.path.join(
                #         modelConfig["test_real_dir"], str(ep) + '_{}.png'.format(i)))

                ''''' save the layout images '''''
                # with open(real_dir + str(b) + '_' + str(it) + '.png', 'rb') as fi:
                #     with PIL.Image.open(fi) as image:
                #         image = transform(image.convert('RGB'))
                #         image = deprocess_fn(image)
                #         image = image.mul(255).clamp(0, 255).byte()
                #         image = image.cpu().detach().numpy()
                #         image = np.transpose(image, (1, 2, 0))
                #         draw_layout(vocab, objs, boxes, save_layout_img_path, str(b) + '_' + str(it) + '.png', masks=None,
                #                     size=32,
                #                     show_boxes=True, bgcolor='white', bg=image)
                it += 1
                if it == 100:
                    b += 1
                    it = 0

        # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # print(sampledImgs2)
        # save_image(sampledImgs2, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledImgName"]+'2.png'), nrow=modelConfig["nrow"])


def eval(modelConfig: Dict):
    from is_score import get_inception_score
    sampled_dir = '/home/lab/Documents/SG-DDPM/TestFakeImgs_wo_guide/'
    real_dir = '/home/lab/Documents/SG-DDPM/TestRealImgs/'
    device = torch.device(modelConfig["device"])
    image_size = (32, 32)
    all_imgs = []
    deprocess_fn = imagenet_deprocess(rescale_image=rescale)
    transforms = [Resize(image_size), T.ToTensor()]
    transform = T.Compose(transforms)
    for filename in os.listdir(sampled_dir):
        f = os.path.join(sampled_dir, filename)
        with open(f, 'rb') as fi:
            with PIL.Image.open(fi) as image:
                WW, HH = image.size
                image = transform(image.convert('RGB'))
                image = deprocess_fn(image)
                image = image.mul(255).clamp(0, 255).byte()
                image = image.cpu().detach().numpy()
                image = np.transpose(image, (1, 2, 0))
                all_imgs.append(image)
    #all_imgs = torch.cat(all_imgs)
    #all_imgs_np = all_imgs.cpu().detach().numpy()
    print('image')
    print(np.max(all_imgs[0]))
    IS, IS_std = get_inception_score(all_imgs)
    # FID = get_fid(all_imgs, real_dir)
    print(IS)
    print(IS_std)
    # print(FID)



    H, W = image_size
    # all_imgs = []
    # for i, (img, objs, boxes, triples) in enumerate(batch):
    #     all_imgs.append(img[None])
    # all_imgs = torch.cat(all_imgs)


def encode_scene_graphs(scene_graphs, vocab, device):
    """
    Encode one or more scene graphs using this model's vocabulary. Inputs to
    this method are scene graphs represented as dictionaries like the following:

    {
      "objects": ["cat", "dog", "sky"],
      "relationships": [
        [0, "next to", 1],
        [0, "beneath", 2],
        [2, "above", 1],
      ]
    }

    This scene graph has three relationshps: cat next to dog, cat beneath sky,
    and sky above dog.

    Inputs:
    - scene_graphs: A dictionary giving a single scene graph, or a list of
      dictionaries giving a sequence of scene graphs.

    Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
    same semantics as self.forward. The returned LongTensors will be on the
    same device as the model parameters.
    """
    if isinstance(scene_graphs, dict):
      # We just got a single scene graph, so promote it to a list
      scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    obj_offset = 0
    for i, sg in enumerate(scene_graphs):
      # Insert dummy __image__ object and __in_image__ relationships
      sg['objects'].append('__image__')
      image_idx = len(sg['objects']) - 1
      for j in range(image_idx):
        sg['relationships'].append([j, '__in_image__', image_idx])

      for obj in sg['objects']:
        obj_idx = vocab['object_name_to_idx'].get(obj, None)
        if obj_idx is None:
          raise ValueError('Object "%s" not in vocab' % obj)
        objs.append(obj_idx)
        obj_to_img.append(i)
      for s, p, o in sg['relationships']:
        pred_idx = vocab['pred_name_to_idx'].get(p, None)
        if pred_idx is None:
          raise ValueError('Relationship "%s" not in vocab' % p)
        triples.append([s + obj_offset, pred_idx, o + obj_offset])
      obj_offset += len(sg['objects'])
    objs = torch.tensor(objs, dtype=torch.int64, device=device)
    triples = torch.tensor(triples, dtype=torch.int64, device=device)
    obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
    print(objs)
    print(triples)
    print(obj_to_img)
    return objs, triples, obj_to_img


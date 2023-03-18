import os

import torch.cuda

from DiffusionFreeGuidence.TrainCondition import train, eval, sample
import torch.multiprocessing as mp


def main(model_config=None):
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 200,
        "batch_size": 1,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition_only_obj/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_0_.pt",
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

    os.makedirs(modelConfig["test_layout_box_dir"], exist_ok=True)
    os.makedirs(modelConfig["test_layout_img_dir"], exist_ok=True)
    os.makedirs(modelConfig["test_sg_dir"], exist_ok=True)
    # os.makedirs(modelConfig["noise_dir"], exist_ok=True)
    os.makedirs(modelConfig["save_dir"], exist_ok=True)
    os.makedirs(modelConfig["test_fake_dir"], exist_ok=True)
    os.makedirs(modelConfig["test_real_dir"], exist_ok=True)

    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        mp.spawn(train, args=(2,), nprocs=modelConfig["world_size"])
        # train(modelConfig)
    elif modelConfig["state"] == "eval":
        eval(modelConfig)
    else:
        sample(modelConfig)


if __name__ == '__main__':
    main()

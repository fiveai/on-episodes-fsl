import os
import datetime
import random
import git
from pprint import PrettyPrinter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import torch.utils.data.distributed

from src.configs import configuration
from src.configs.load_yaml import load_dataset_yaml
from src.configs.get_configs import get_dataloader, get_scheduler, get_optimizer
from src.utils.meters import warp_tqdm
from src.utils.logs import log_experiment, save_checkpoint
from src.train.loss import FewShotNCALoss, SupervisedContrastiveLoss
from src.train.train import train
from src.utils.evaluation import (
    extract_and_evaluate,
    extract_and_evaluate_allshots,
    validate_loss,
)
from src.utils.meters import BestAccuracySlots
from src.train.opt_supportset import optimize_full_model_episodic
import src.models as models


def main():
    args = configuration.parser_args()

    num_classes, args.data, args.split_dir = load_dataset_yaml(args.dataset)

    # num_classes determined by load_dataset_yaml, except if set manually
    # done in cases when doing cross domain experiments
    if not args.num_classes:
        args.num_classes = num_classes

    repo = git.Repo(search_parent_directories=True)

    #assert not repo.is_dirty(
    #    untracked_files=False), 'Please commit your changes before running any experiment (comment this code if you want to run experiments without commiting).'

    now = datetime.datetime.now()
    datetime_string = now.strftime("%y-%m-%d_%H-%M-%S")

    # print options as dictionary and save to output
    PrettyPrinter(indent=4).pprint(vars(args))

    # generate some warning messages for arguments which are incompatible
    if args.xent_weight > 0 and args.proto_train:
        raise NotImplementedError(
            "\n>> Cannot train a prototypical network simultaneously with standard crossentropy classification "
        )

    # log experiment parameters and commit sha for reproducability
    log_experiment(repo, datetime_string, args)

    # if seed is set, run will be deterministic for reproducability
    if args.seed is not None:
        print("\n>> Using fixed seed #" + str(args.seed))
        # Not fully deterministic, but without cudnn.benchmark is slower
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        cudnn.benchmark = True

    expm_id = datetime_string + args.expm_id
    tb_writer_train = SummaryWriter("../runs/" + expm_id + "/train/")
    tb_writer_val = SummaryWriter("../runs/" + expm_id + "/val/")

    # init meter to store best values
    best_accuracy_meter = BestAccuracySlots()

    # create model
    model = models.__dict__[args.arch](
        feature_dim=args.projection_feat_dim,
        num_classes=args.num_classes,
        projection=args.projection,
        use_fc=args.xent_weight > 0 or args.pretrained_model,
    )

    model = torch.nn.DataParallel(model).cuda()

    # define xent loss function (criterion) and optimizer
    xent = nn.CrossEntropyLoss().cuda()
    print("\n>> Number of CUDA devices: " + str(torch.cuda.device_count()))

    # either choose contrastive loss or
    if args.contrastiveloss:
        # use supervised contrastive loss
        loss = SupervisedContrastiveLoss(
            args.num_classes, batch_size=args.batch_size, temperature=args.temperature
        ).cuda()
        loss_norm = loss
    else:
        loss = FewShotNCALoss(
            args.num_classes,
            batch_size=args.batch_size,
            temperature=args.temperature,
            frac_negative_samples=args.negatives_frac_random,
            frac_positive_samples=args.positives_frac_random,
        ).cuda()
        # loss_norm used for computing validation NCA loss
        loss_norm = FewShotNCALoss(
            args.num_classes,
            batch_size=args.batch_size,
            temperature=args.temperature,
            frac_negative_samples=1,
            frac_positive_samples=1,
        ).cuda()

    # train loader is different when training protonets, due to batch creation
    if args.proto_train:
        sample_info = [
            args.proto_train_iter,
            args.proto_train_way,
            args.proto_train_shot,
            args.proto_train_query,
        ]
        train_loader = get_dataloader(
            "train", args, not args.disable_train_augment, sample=sample_info
        )
    else:
        train_loader = get_dataloader(
            "train", args, not args.disable_train_augment, shuffle=True
        )

    # init train loader used for centering
    train_loader_for_avg = get_dataloader(
        "train", args, aug=False, shuffle=False, out_name=False
    )

    # init standard validation and test loader
    val_loader = get_dataloader("val", args, aug=False, shuffle=False, out_name=False)
    test_loader = get_dataloader("test", args, aug=False, shuffle=False, out_name=False)

    # init optimizer and scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(len(train_loader), optimizer, args)

    if args.resume_model:
        if os.path.isfile(args.resume_model):
            checkpoint = torch.load(args.resume_model)
            args.start_epoch = checkpoint["epoch"]
            best_accuracy_meter = checkpoint["best_accuracies"]
            scheduler.load_state_dict(checkpoint["scheduler"])
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "\n>> Resume training for previously trained model (epoch %d)"
                % args.start_epoch
            )
        else:
            raise NameError("Invalid path name:{}".format(args.resume_model))

    elif args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            checkpoint = torch.load(args.pretrained_model)
            model.load_state_dict(checkpoint["state_dict"])
            print("\n>> Loaded pretrained model")
        else:
            raise NameError("Invalid path name:{}".format(args.resume_model))

    if args.episode_optimize:
        sample_info = [
            args.proto_train_iter,
            args.proto_train_way,
            args.proto_train_shot,
            args.proto_train_query,
        ]
        episodic_test_loader = get_dataloader(
            "test", args, aug=False, sample=sample_info, shuffle=False, out_name=False
        )
        optimize_full_model_episodic(
            episodic_test_loader, train_loader, model, loss, args
        )

    # evaluate a specific model.
    if args.evaluate_model:
        print("\n>> Evaluating previously trained model on the test set")
        if args.evaluate_all_shots:
            extract_and_evaluate_allshots(
                model,
                train_loader_for_avg,
                test_loader,
                "test",
                args,
                writer=None,
                model_name=args.evaluate_model,
                expm_id=expm_id,
                t=0,
                print_stdout=True,
            )
        else:
            print("\n>>> Evaluating Test Set")
            extract_and_evaluate(
                model,
                train_loader_for_avg,
                test_loader,
                "test",
                args,
                writer=None,
                model_name=args.evaluate_model,
                expm_id=expm_id,
                t=0,
                print_stdout=True,
            )
            print("\n>>> Evaluating Validation Set")
            extract_and_evaluate(
                model,
                train_loader_for_avg,
                val_loader,
                "val",
                args,
                writer=None,
                model_name=args.evaluate_model,
                expm_id=expm_id,
                t=0,
                print_stdout=True,
            )
        return

    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)), args)
    for epoch in tqdm_loop:

        # train for one epoch
        train(
            train_loader,
            model,
            xent,
            loss,
            optimizer,
            epoch,
            scheduler,
            tb_writer_train,
            args,
        )
        scheduler.step(epoch)

        # evaluate on meta validation set
        if (epoch + 1) % args.val_interval == 0:
            # compute loss on the validation set
            nca_loss_val, xent_loss_val = validate_loss(val_loader, model, loss_norm, xent, args)
            tb_writer_val.add_scalar("Loss/SoftNN", nca_loss_val, epoch * len(train_loader))
            if args.xent_weight > 0:
                tb_writer_val.add_scalar(
                    "Loss/X-entropy", xent_loss_val, epoch * len(train_loader)
                )
            # full evaluation on the val set
            shot1_info, shot5_info = extract_and_evaluate(
                model,
                train_loader_for_avg,
                val_loader,
                "val",
                args,
                writer=tb_writer_val,
                t=epoch * len(train_loader),
            )
            # update best accuracies
            is_best1, is_best5 = best_accuracy_meter.update(shot1_info, shot5_info)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "scheduler": scheduler.state_dict(),
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_accuracies": best_accuracy_meter,
                    "optimizer": optimizer.state_dict(),
                },
                is_best1,
                is_best5,
                folder=args.save_path,
                filename=expm_id,
            )

    # print the best epoch accuracies for the val set
    tb_writer_val.add_scalar("val/1-shot/Best-CL2N", best_accuracy_meter.cl2n_1shot, 0)
    tb_writer_val.add_scalar("val/5-shot/Best-CL2N", best_accuracy_meter.cl2n_5shot, 0)

    # at the end of training, evaluate on the VAL set with the best performing epoch of the model just trained
    # on more epochs than during training
    extract_and_evaluate(
        model,
        train_loader_for_avg,
        val_loader,
        "val",
        args,
        model_name=expm_id,
        writer=tb_writer_val,
        t=0,
        print_stdout=True,
        expm_id=expm_id,
        num_iter=args.test_iter # set number of iter to same as test iter (higher than val iter)
    )

    # at the end of training, evaluate on the TEST set with the best performing epoch of the model just trained
    extract_and_evaluate(
        model,
        train_loader_for_avg,
        test_loader,
        "test",
        args,
        model_name=expm_id,
        writer=tb_writer_val,
        t=0,
        print_stdout=True,
        expm_id=expm_id
    )

    tb_writer_train.close()
    tb_writer_val.close()


if __name__ == "__main__":
    main()

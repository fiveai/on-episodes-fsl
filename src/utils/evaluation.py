import torch
from numpy import linalg as LA
import random
from scipy.stats import mode
import numpy as np
import collections
from src.configs.get_configs import get_metric
from src.utils.meters import AverageMeter, warp_tqdm, compute_confidence_interval


def metric_prediction(gallery, query, train_label, metric_type, args):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type, args)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)
    return predict


def metric_class_type(
    gallery,
    query,
    train_label,
    test_label,
    shot,
    args,
    out_cov,
    train_mean=None,
    norm_type="CL2N",
):
    """
    this function performs test-time classification
    """

    # generate the classification space according to normalisation
    if norm_type == "CL2N":
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == "L2N":
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    # if we don't do soft assignment, and no nearest neighbour (num_NN = 1), then compute
    # the prototypes
    if (args.num_NN == 1 or shot == 1) and not args.soft_assignment:
        if args.median_prototype:
            gallery = np.median(
                gallery.reshape(args.test_way, shot, gallery.shape[-1]), axis=1
            )
        else:
            gallery = gallery.reshape(args.test_way, shot, gallery.shape[-1]).mean(
                1
            )
        train_label = train_label[::shot]
        num_NN = 1
    else:
        num_NN = args.num_NN

    # If we do evaluation with soft assignment we evaluate differently.
    if args.soft_assignment:
        subtract = gallery[:, None, :] - query
        distance = np.exp(-1 * np.sum(subtract ** 2, axis=2))
        norm_distance = distance / distance.sum(0)[None, :]

        # we can do kNN if number of shots > 1
        if num_NN > 1 and shot != 1:
            # get the closest labels
            idx = np.argsort(norm_distance, axis=0)[-num_NN:]
            nearest_samples_labels = np.take(train_label, idx)
            nearest_samples_distances = np.sort(norm_distance, axis=0)[-num_NN:]

            weighted_nearest_neighbours = np.zeros(
                (args.num_classes, nearest_samples_labels.shape[-1])
            )
            # sum the total contribution of each example
            for i, row in enumerate(nearest_samples_labels):
                for j, element in enumerate(row):
                    weighted_nearest_neighbours[
                        element, j
                    ] += nearest_samples_distances[i, j]

            # predict
            predictions = np.argmax(weighted_nearest_neighbours, axis=0)
            acc = (predictions == test_label).mean()
            return acc
        else:
            # reshape the norm_distances and sum the likelihoods to get likelihood for each class
            norm_distance = norm_distance.reshape(
                args.test_way, shot, norm_distance.shape[-1]
            ).sum(1)
            # get the train labels
            train_label = train_label[::shot]

            # get predictions
            prediction_idx = np.argmax(norm_distance, axis=0)
            predictions = np.take(train_label, prediction_idx)
            acc = (predictions == test_label).mean()
            return acc
    else:
        # if not doing soft assignment, just compute the nearest neighbors
        subtract = gallery[:, None, :] - query
        distance = LA.norm(subtract, 2, axis=-1)
        idx = np.argpartition(distance, num_NN, axis=0)[:num_NN]
        nearest_samples = np.take(train_label, idx)
        out = mode(nearest_samples, axis=0)[0]
        out = out.astype(int)
        test_label = np.array(test_label)
        acc = (out == test_label).mean()
        return acc


def meta_evaluate(data, train_mean, out_cov, shot, num_iter, args, expm_id=None, split=""):
    cl2n_list = []
    for _ in warp_tqdm(range(num_iter), args):
        # record accuracies for all three types of normalisation
        train_data, test_data, train_label, test_label = sample_case(data, shot, args)
        acc = metric_class_type(
            train_data,
            test_data,
            train_label,
            test_label,
            shot,
            args,
            out_cov,
            train_mean=train_mean,
            norm_type="CL2N",
        )
        cl2n_list.append(acc)

    # save
    if expm_id:
        np.save(args.save_path + "/numpy_results/" + expm_id + "_" + str(split) + "_shot" + str(shot) + ".npy", np.array(cl2n_list))
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return cl2n_mean, cl2n_conf


def extract_feature(train_loader, val_loader, model, args, tag="last"):

    # We use the FC layer of the model only if we are combining XENT with NCA loss
    use_fc = args.xent_weight > 0
    print("\n>> Extracting statistics from training set embeddings")
    model.eval()
    with torch.no_grad():
        # get training mean
        out_mean, fc_out_mean, out_cov = [], [], []
        for i, (inputs, _) in enumerate(warp_tqdm(train_loader, args)):
            outputs, fc_outputs = model(
                inputs, use_fc=use_fc, cat=args.multi_layer_eval
            )
            outputs_np = outputs.cpu().data.numpy()
            out_mean.append(outputs_np)
            out_cov.append(outputs_np)
            if fc_outputs is not None:
                fc_out_mean.append(fc_outputs.cpu().data.numpy())
        out_mean = np.concatenate(out_mean, axis=0).mean(0)
        out_cov = np.cov(np.concatenate(out_cov, axis=0).T)

        if len(fc_out_mean) > 0:
            fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
        else:
            fc_out_mean = -1

        output_dict = collections.defaultdict(list)
        fc_output_dict = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(warp_tqdm(val_loader, args)):
            # compute output
            outputs, fc_outputs = model(
                inputs, use_fc=use_fc, cat=args.multi_layer_eval
            )
            outputs = outputs.cpu().data.numpy()
            if fc_outputs is not None:
                fc_outputs = fc_outputs.cpu().data.numpy()
            else:
                fc_outputs = [None] * outputs.shape[0]
            for out, fc_out, label in zip(outputs, fc_outputs, labels):
                output_dict[label.item()].append(out)
                fc_output_dict[label.item()].append(fc_out)
        all_info = [out_mean, fc_out_mean, output_dict, fc_output_dict, out_cov]
        # save_pickle(save_dir + '/output.plk', all_info)
        return all_info


def extract_and_evaluate_allshots(
    model,
    train_loader_for_avg,
    eval_loader,
    split,
    args,
    model_name=None,
    writer=None,
    t=None,
    print_stdout=False,
):
    """
    This function evaluates the last model performing model after training it.
    from 1 to k shots, where k is specified by args.evaluate_all_shots. Prints
    the scores, and saves them under args.save_id
    """

    num_iter = args.test_iter if split == "test" else args.val_iter

    checkpoint1 = torch.load("{}_checkpoint.pth.tar".format(model_name))
    model.load_state_dict(checkpoint1["state_dict"])
    # compute training dataset statistics
    out_mean, fc_out_mean, out_dict, fc_out_dict, out_cov = extract_feature(
        train_loader_for_avg, eval_loader, model, args
    )

    save_shot_npy = np.zeros(args.evaluate_all_shots)
    for k in range(1, args.evaluate_all_shots + 1):
        shot_info = tuple(
            [
                100 * x
                for x in meta_evaluate(out_dict, out_mean, out_cov, k, num_iter, args)
            ]
        )
        save_shot_npy[k - 1] = shot_info[0]

        print(
            ">>>\t ### {} set:\nfeature\tUN\tL2N\tCL2N\n{}\t{:2.2f}({:2.2f})\t{:2.2f}({:2.2f})\t{:2.2f}({:2.2f})".format(
                split, "GVP " + str(k) + " Shot", *shot_info
            )
        )

    print("\n >>> Array being saved:", save_shot_npy)
    np.save(args.save_id, save_shot_npy)


def extract_and_evaluate(
    model,
    train_loader_for_avg,
    eval_loader,
    split,
    args,
    model_name=None,
    writer=None,
    t=None,
    print_stdout=False,
    expm_id=None,
    num_iter = None
):
    """
    This function evaluates the best and last performing model after training it.
    arguments are the model, the tensorboard writer and the expm_id.
    Prints the 1-shot and 5-shot scores for best and last model.
    """

    if not num_iter:
        num_iter = args.test_iter if split == "test" else args.val_iter

    if model_name:
        checkpoint1 = torch.load(
            "{}/{}_best1.pth.tar".format(args.save_path, model_name)
        )
        model.load_state_dict(checkpoint1["state_dict"])
        out_mean1, fc_out_mean1, out_dict1, fc_out_dict1, out_cov1 = extract_feature(
            train_loader_for_avg, eval_loader, model, args
        )
        checkpoint5 = torch.load(
            "{}/{}_best5.pth.tar".format(args.save_path, model_name)
        )
        model.load_state_dict(checkpoint5["state_dict"])
        out_mean5, fc_out_mean5, out_dict5, fc_out_dict5, out_cov5 = extract_feature(
            train_loader_for_avg, eval_loader, model, args
        )
    else:
        # compute training dataset statistics
        out_mean1, fc_out_mean1, out_dict1, fc_out_dict1, out_cov1 = extract_feature(
            train_loader_for_avg, eval_loader, model, args
        )
        # When model_name is not passed, we are using the current model checkpoint, which is the same for 1-shot and 5-shot
        out_mean5, fc_out_mean5, out_dict5, fc_out_dict5, out_cov5 = (
            out_mean1,
            fc_out_mean1,
            out_dict1,
            fc_out_dict1,
            out_cov1,
        )

    shot1_info = tuple(
        [
            100 * x
            for x in meta_evaluate(out_dict1, out_mean1, out_cov1, 1, num_iter, args, expm_id, split)
        ]
    )
    shot5_info = tuple(
        [
            100 * x
            for x in meta_evaluate(out_dict5, out_mean5, out_cov5, 5, num_iter, args, expm_id, split)
        ]
    )

    if print_stdout:
        print(
            ">>>\t ### {} set:\nfeature\tCL2N\n{}\t{:2.2f}({:2.2f})\n{}\t{:2.2f}({:2.2f})".format(
                split, "GVP 1Shot", *shot1_info, "GVP_5Shot", *shot5_info
            )
        )

    if writer:
        writer.add_scalar(split + "/1-shot/CL2N", shot1_info[0], t)
        writer.add_scalar(split + "/5-shot/CL2N", shot5_info[0], t)

    return shot1_info, shot5_info


def sample_case(ld_dict, shot, args):
    sample_class = random.sample(list(ld_dict.keys()), args.test_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + args.test_query)
        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label


def validate_loss(val_loader, model, nca_f, xent_f, args):
    """
    input:
    :param: val loader. DataLoader with validation images loaded.
    :model: neural network model
    :nca_f: NCA criterion
    :xent_f: cross-entropy criterion
    :args: arguments passed from command line
    """
    nca_losses, xent_losses = AverageMeter(), AverageMeter()
    model.eval()

    tqdm_val_loader = warp_tqdm(val_loader, args)
    with torch.no_grad():
        for input, target in tqdm_val_loader:
            if args.xent_weight > 0:
                features, fc_output = model(input, use_fc=True)
                xent_loss = xent_f(fc_output, target.cuda(non_blocking=True))
                xent_losses.update(xent_loss)
            else:
                features, _ = model(input, use_fc=False)

            nca_loss_val = nca_f(features, target)
            nca_losses.update(nca_loss_val.item(), input.size(0))
            tqdm_val_loader.set_description(
                "NCA loss (val): {:.2f}".format(nca_losses.avg)
            )

    return nca_losses.avg, xent_losses.avg

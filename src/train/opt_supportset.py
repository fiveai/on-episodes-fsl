import copy
import torch
from src.utils.evaluation import warp_tqdm, AverageMeter


def optimize_full_model_episodic(episodic_test_loader, train_loader, model, nca_l, args):
    model.eval()
    with torch.no_grad():
        # get training mean
        outputs = []
        targets = []
        for i, (inputs, target) in enumerate(warp_tqdm(train_loader, args)):
            embedding, fc_outputs = model(
                inputs, use_fc=False, cat=args.multi_layer_eval
            )
            outputs.append(embedding)
            targets.append(target)

    outputs_torch_train = torch.cat(outputs, axis=0)
    mean_output = torch.mean(outputs_torch_train, axis=0)

    tqdm_test_loader = warp_tqdm(episodic_test_loader, args)
    n_epochs = 2

    accuracy_meter = AverageMeter()
    for i, (input, target) in enumerate(tqdm_test_loader):
        model_copy = copy.deepcopy(model).cuda()
        model_copy.train()
        optimizer = torch.optim.Adam(
            model_copy.parameters(), lr=0.0001, weight_decay=5e-4
        )
        for i in range(n_epochs):
            features, output = model_copy(input, use_fc=False)

            # first part of batch is support set
            support_feature = features[: args.proto_train_shot * args.proto_train_way]
            support_target = target[: args.proto_train_shot * args.proto_train_way]

            # the rest for computing the query set
            query_feature = features[args.proto_train_shot * args.proto_train_way :]
            query_target = target[args.proto_train_shot * args.proto_train_way :]

            loss = nca_l(support_feature, support_target).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_copy.eval()
        with torch.no_grad():
            features, output = model_copy(
                input, use_fc=False, cat=args.multi_layer_eval
            )
            features = features - mean_output
            features = features / torch.norm(features, 2, 1)[:, None]
            support_feature = features[: args.proto_train_shot * args.proto_train_way]
            support_target = target[: args.proto_train_shot * args.proto_train_way]

            # the rest for computing the query set
            query_feature = features[args.proto_train_shot * args.proto_train_way :]
            # query_feature = query_feature / torch.norm(query_feature)
            query_target = target[args.proto_train_shot * args.proto_train_way :]

            # compute the PROTOTYPICAL NETWORK loss
            n = args.proto_train_way * args.proto_train_query
            support_feature = support_feature.reshape(
                args.proto_train_way, args.proto_train_shot, -1
            ).mean(1)

            support_target = target[: args.proto_train_shot * args.proto_train_way][
                0 :: args.proto_train_shot
            ]

            p_norm = torch.cdist(support_feature, query_feature, p=2).cuda()

            # create pre
            predictions_idx = torch.argmin(p_norm, axis=0)
            predictions = support_target[predictions_idx]

            accuracy = torch.sum(predictions == query_target).cpu().data.numpy() / n
            accuracy_meter.update(accuracy)
            print(accuracy_meter.avg * 100)

    return

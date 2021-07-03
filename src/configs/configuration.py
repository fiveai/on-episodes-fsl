import configargparse


def parser_args():
    parser = configargparse.ArgParser()
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', default='miniimagenet', choices=('miniimagenet', 'tieredimagenet', 'CIFARFS'))
    parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num-classes', default=0, type=int, metavar='N',
                        help='number of classes in the dataset')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--disable-train-augment', action='store_true',
                        help='disable training augmentation')
    parser.add_argument('--disable-random-resize', action='store_true',
                        help='disable random resizing')
    parser.add_argument('--enlarge', action='store_true',
                        help='enlarge the image size then center crop')
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet12',
                        help='network architecture')
    parser.add_argument('--scheduler', default='multi_step', choices=('step', 'multi_step'),
                        help='scheduler, the detail is shown in train.py')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-stepsize', default=30, type=int,
                        help='learning rate decay step size (for "step" scheduler)')
    parser.add_argument('--lr-milestones', default="0.7",
                        help='Fractions of total epochs at which decay the LR (for "multi_step" scheduler). Seperate values using commas ,')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'Adam'))
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov for SGD, disable it in default')
    parser.add_argument('--optimize-temperature', action='store_true',
                        help='optimize temperature parameter of NCA using SGD during training')

    parser.add_argument('--projection', action='store_true',
                        help='Use projection network')
    parser.add_argument('--projection-feat-dim', type=int, default=512,
                        help='Feature dimensionality of output of projection network')

    parser.add_argument('--multi-layer-eval', action='store_true',
                        help='Use earlier layers as embedding during evalution')
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--test-iter', type=int, default=10000,
                        help='number of iterations on test set')
    parser.add_argument('--val-iter', type=int, default=3000,
                        help='number of iterations on val set')
    parser.add_argument('--test-way', type=int, default=5,
                        help='number of ways during val/test')
    parser.add_argument('--test-query', type=int, default=15,
                        help='number of queries during val/test')
    parser.add_argument('--val-interval', type=int, default=10,
                        help='evaluate every X epoch epochs')
    parser.add_argument('--num-NN', type=int, default=1,
                        help='number of nearest neighbors, set this number >1 when do kNN')
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--expm-id', default='test', type=str,
                        help='experiment name for logging rsults')
    parser.add_argument('--save-id', default='', type=str,
                        help='argument for a save file to save results for --evaluate-all-shots')
    parser.add_argument('--save-path', default='../results', type=str,
                        help='path to folder stored the log and checkpoint')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-tqdm', action='store_true',
                        help='disable tqdm.')
    parser.add_argument('--soft-assignment', action='store_true',
                        help='use soft assignment for multiple shot classification.')
    parser.add_argument('--contrastiveloss', action='store_true',
                        help='Used the supervised contrastive loss instead of NCA')
    parser.add_argument('--replacement-sampling', action='store_true',
                        help='for non-episodic batch generation, sample with replacement (num. batches needs to be set by proto-train-iter argument)')
    parser.add_argument('--resume-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint of the model you want to resume training')
    parser.add_argument('--episode-no-replacement-sampling', action='store_false',
                        help='for episode batch generation, sample batches with replacement by default, if triggered episodes are sampled without replacement')
    parser.add_argument('--evaluate-model', default='', type=str, metavar='PATH',
                        help='path to the model to evaluate on test set')
    parser.add_argument('--evaluate-all-shots', type=int, default=0,
                        help='do evaluation over multiple values of shot')
    parser.add_argument('--xent-weight', default=0, type=float,
                        help='part of the loss that should consist of cross entropy training')
    parser.add_argument('--negatives-frac-random', default=1, type=float,
                        help='fraction of random negatives to sample for the NCA loss (default: 1)')
    parser.add_argument('--positives-frac-random', default=1, type=float,
                        help='fraction of random negatives to sample for the NCA loss (default: 1)')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='path to the pretrained model')

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--proto-train', action='store_true',
                        help='do prototypical training')
    parser.add_argument('--proto-train-iter', type=int, default=100,
                        help='number of iterations for proto train')
    parser.add_argument('--proto-train-way', type=int, default=30,
                        help='number of ways for protonet training')
    parser.add_argument('--proto-train-shot', type=int, default=1,
                        help='number of shots for protonet training')
    parser.add_argument('--proto-train-query', type=int, default=15,
                        help='number of queries for protonet training')
    parser.add_argument('--proto-disable-aggregates', action='store_true',
                        help='disables the construction of aggregates in prototypical networks')
    parser.add_argument('--proto-enable-all-pairs', action='store_true',
                        help='disregards the split between the query and support set and computes all pairs of distances')
    parser.add_argument('--median-prototype', action='store_true',
                        help='use median instead of mean for computing prototypes')
    parser.add_argument('--episode-optimize', action='store_true',
                        help='optimize model on support set during evaluation')

    return parser.parse_args()

from __future__ import print_function
import sys, os, pdb
sys.path.insert(0, 'src')
import numpy as np, scipy.misc 
from optimize_with_multiple_checkpoints import optimize
from argparse import ArgumentParser
from utils import save_img, get_img, exists, list_files
import evaluate
import time
from datetime import datetime


CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 4
# DEVICE = '/gpu:0'
FRAC_GPU = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    
    parser.add_argument('--device', type=str,
                        dest='device_and_number',
                        help='device and its associated GPU number or CPU count',
                        metavar='DEVICE', required=True)
    
    parser.add_argument('--max-runtime-in-minutes', type=int,
                        dest='max_runtime_in_minutes',
                        help='maximum runtime in minutes before automatic shut down',
                        metavar='MAX_RUNTIME_IN_MINUTES', default=float('inf'))

    parser.add_argument('--log_file', type=str,
                        dest='log_file',
                        help='file you want to write the print statements to',
                        metavar='LOG_FILE', default=None)

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0
    assert opts.max_runtime_in_minutes >= 0
    if opts.log_file:
        assert os.path.isdir(os.path.dirname(opts.log_file))

def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]

    
def main():
    parser = build_parser()
    options = parser.parse_args()
    _run_model(options)


def _run_model(options):
    check_opts(options)
    style_target = get_img(options.style)
    if not options.slow:
        content_targets = _get_files(options.train_path)
    elif options.test:
        content_targets = [options.test]

    kwargs = {
        "slow":options.slow,
        "epochs":options.epochs,
        "print_iterations":options.checkpoint_iterations,
        "batch_size":options.batch_size,
        "save_path":os.path.join(options.checkpoint_dir,'fns.ckpt'),
        "learning_rate":options.learning_rate,
        "device_and_number":options.device_and_number
    }

    if options.slow:
        if options.epochs < 10:
            kwargs['epochs'] = 1000
        if options.learning_rate < 1:
            kwargs['learning_rate'] = 1e1

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    start_time = time.time()
    shutdown_time = start_time +  options.max_runtime_in_minutes * 60
    if options.log_file:
        log_file_ = open(options.log_file, 'w')
        sys.stdout = log_file_
        sys.stderr = log_file_
    for preds, losses, i, epoch, checkpoint_number in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses
        delta_time, start_time = time.time() - start_time, time.time()        
        print('Current Time = {}; Time Elapsed = {}; Epoch = {}; Iteration = {}; Loss = {}'.format(
            datetime.now().strftime("%Y %B %d, %H:%M:%S"), delta_time, epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('Loss values: style = %s; content = %s; tv = %s' % to_print)
        sys.stdout.flush()
        if options.test:
            assert options.test_dir != False
            preds_path = '%s/%s_%s.png' % (options.test_dir,epoch,i)
            if not options.slow: # if uses GPU, uses RAM that it doesn't have, so it's slow here
                # ckpt_dir = os.path.dirname(options.checkpoint_dir)
                actual_dir = os.path.join(options.checkpoint_dir, 'checkpoint_{}'.format(checkpoint_number))
                evaluate.ffwd_to_img(options.test, preds_path, actual_dir)
            else:
                save_img(preds_path, img)
        if time.time() > shutdown_time: # automatic shut down
            print('Ran for maximum runtime in minutes. Now shutting down time.')
            break
    ckpt_dir = options.checkpoint_dir
    cmd_text = 'python evaluate.py --checkpoint %s ...' % ckpt_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)
    if options.log_file:
        log_file_.close()
    

class ArgumentObject(object):
    def __init__(self, style, checkpoint_dir, device_and_number, test, test_dir, 
                 epochs, checkpoint_iterations, batch_size, max_runtime_in_minutes,
                 log_file, train_path, vgg_path, content_weight, style_weight, 
                 tv_weight, learning_rate, slow):
        self.style = style
        self.checkpoint_dir = checkpoint_dir
        self.device_and_number = device_and_number
        self.test = test
        self.test_dir = test_dir
        self.epochs = epochs
        self.checkpoint_iterations = checkpoint_iterations
        self.batch_size = batch_size
        self.max_runtime_in_minutes = max_runtime_in_minutes
        self.log_file = log_file
        self.train_path = train_path
        self.vgg_path = vgg_path
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.learning_rate = learning_rate
        self.slow = slow
        
def create_checkpoints(
        # mandatory arguments
        style, checkpoint_dir, device_and_number,

        # useful argments
        test=False, test_dir=False, epochs=NUM_EPOCHS, 
        checkpoint_iterations=CHECKPOINT_ITERATIONS, batch_size=BATCH_SIZE,
        max_runtime_in_minutes=float('inf'),
        log_file=None,

        # not useful arguments you probably shouldn't touch
        train_path=TRAIN_PATH, vgg_path=VGG_PATH, content_weight=CONTENT_WEIGHT, 
        style_weight=STYLE_WEIGHT, tv_weight=TV_WEIGHT, learning_rate=LEARNING_RATE, slow=False
):
    options = ArgumentObject(style, checkpoint_dir, device_and_number, test, 
            test_dir, epochs, checkpoint_iterations, batch_size, 
             max_runtime_in_minutes, log_file, train_path, vgg_path, 
             content_weight, style_weight, tv_weight, learning_rate, slow)
    _run_model(options)
    
    
    
    
if __name__ == '__main__':
    main()

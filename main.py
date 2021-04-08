#!/usr/env/bin python3

"""
Generate training and test images.
"""
import argparse
import multiprocessing as mp
import os
import traceback
from itertools import repeat

import cv2
import numpy as np
from tenacity import retry

import libs.font_utils as font_utils
import libs.utils as utils
from libs.config import load_config
from libs.timer import Timer
from textrenderer.corpus.corpus_utils import corpus_factory
from textrenderer.renderer import Renderer

# prevent opencv use all cpus
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_img', type=int, default=20,
                        help="Number of images to generate")

    parser.add_argument('--length', type=int, default=10,
                        help='The max length of Chars(chn) or words(eng) '
                        'in a image. For eng corpus mode, default length is 3')

    parser.add_argument('--clip_max_chars', action='store_true', default=False,
                        help='For training a CRNN model, max number of chars in an image'
                             'should less then the width of last CNN layer.')

    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=256,
                        help="If 0, output images will have different width")

    parser.add_argument('--chars_file', type=str, default='./data/chars/chn.txt',
                        help='Chars allowed to be appear in generated images.')

    parser.add_argument('--config_file', type=str, default='./configs/default.yaml',
                        help='Set the parameters when rendering images')

    parser.add_argument('--fonts_list', type=str, default='./data/fonts_list/chn.txt',
                        help='Fonts file path to use')

    parser.add_argument('--bg_dir', type=str, default='./data/bg',
                        help="Some text images(according to your config in yaml file) will"
                             "use pictures in this folder as background")

    parser.add_argument('--corpus_dir', type=str, default="./data/corpus",
                        help='When corpus_mode is chn or eng, text on image will randomly selected from corpus.'
                             'Recursively find all txt file in corpus_dir')

    parser.add_argument('--corpus_mode', type=str, default='chn', choices=['random', 'chn', 'eng', 'list'],
                        help='Different corpus type have different load/get_sample method'
                             'random: random pick chars from chars file'
                             'chn: pick continuous chars from corpus'
                             'eng: pick continuous words from corpus, space is included in label')

    parser.add_argument('--output_dir', type=str,
                        default='./output', help='Images save dir')

    parser.add_argument('--tag', type=str, default='default',
                        help='output images are saved under output_dir/{tag} dir')

    parser.add_argument('--debug', action='store_true',
                        default=False, help="output uncroped image")

    parser.add_argument('--viz', action='store_true', default=False)

    parser.add_argument('--strict', action='store_true', default=False,
                        help="check font supported chars when generating images")

    parser.add_argument('--gpu', action='store_true',
                        default=False, help="use CUDA to generate image")

    parser.add_argument('--num_processes', type=int, default=None,
                        help="Number of processes to generate image. If None, use all cpu cores")

    flags, _ = parser.parse_known_args()
    flags.save_dir = os.path.join(flags.output_dir, flags.tag)

    if os.path.exists(flags.bg_dir):
        num_bg = len(os.listdir(flags.bg_dir))
        flags.num_bg = num_bg

    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    if flags.num_processes == 1:
        parser.error("num_processes min value is 2")

    return flags


lock = mp.Lock()
counter = mp.Value('i', 0)
STOP_TOKEN = 'kill'

flags = parse_args()
cfg = load_config(flags.config_file)

fonts = font_utils.get_font_paths_from_list(flags.fonts_list)
bgs = utils.load_bgs(flags.bg_dir)

corpus = corpus_factory(flags.corpus_mode, flags.chars_file,
                        flags.corpus_dir, flags.length)

renderer = Renderer(corpus, fonts, bgs, cfg,
                    height=flags.img_height,
                    width=flags.img_width,
                    clip_max_chars=flags.clip_max_chars,
                    debug=flags.debug,
                    gpu=flags.gpu,
                    strict=flags.strict)


def start_listen(q, fname):
    """ listens for messages on the q, writes to file. """

    f = open(fname, mode='a', encoding='utf-8')
    while 1:
        m = q.get()
        if m == STOP_TOKEN:
            break
        try:
            f.write(str(m) + '\n')
        except:
            traceback.print_exc()

        with lock:
            if counter.value % 1000 == 0:
                f.flush()
    f.close()


@retry
def gen_img_retry(renderer, img_index):
    try:
        return renderer.gen_img(img_index)
    except Exception as e:
        print("Retry gen_img: %s" % str(e))
        traceback.print_exc()
        raise Exception


def generate_img(img_index, q=None):
    global flags, lock, counter
    # Make sure different process has different random seed
    np.random.seed()

    im, word = gen_img_retry(renderer, img_index)

    base_name = '{:08d}'.format(img_index)

    if not flags.viz:
        fname = os.path.join(flags.save_dir, base_name + '.jpg')
        cv2.imwrite(fname, im)

        label = "{} {}".format(base_name, word)

        if q is not None:
            q.put(label)

        with lock:
            counter.value += 1
            print_end = '\n' if counter.value == flags.num_img else '\r'
            if counter.value % 100 == 0 or counter.value == flags.num_img:
                print("{}/{} {:2d}%".format(counter.value,
                                            flags.num_img,
                                            int(counter.value / flags.num_img * 100)),
                      end=print_end)
    else:
        utils.viz_img(im)


def sort_labels(tmp_label_fname, label_fname):
    lines = []
    with open(tmp_label_fname, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = sorted(lines)
    with open(label_fname, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line[9:])


def restore_exist_labels(label_path):
    # 如果目标目录存在 labels.txt 则向该目录中追加图片
    start_index = 0
    if os.path.exists(label_path):
        start_index = len(utils.load_chars(label_path))
        print('Generate more text images in %s. Start index %d' %
              (flags.save_dir, start_index))
    else:
        print('Generate text images in %s' % flags.save_dir)
    return start_index


def get_num_processes(flags):
    processes = flags.num_processes
    if processes is None:
        processes = max(os.cpu_count(), 2)
    return processes


if __name__ == "__main__":
    # It seems there are some problems when using opencv in multiprocessing fork way
    # https://github.com/opencv/opencv/issues/5150#issuecomment-161371095
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-382660636
    if utils.get_platform() == "OS X":
        mp.set_start_method('spawn', force=True)

    if flags.viz == 1:
        flags.num_processes = 1

    tmp_label_path = os.path.join(flags.save_dir, 'tmp_labels.txt')
    label_path = os.path.join(flags.save_dir, 'labels.txt')

    manager = mp.Manager()
    q = manager.Queue()

    start_index = restore_exist_labels(label_path)

    timer = Timer(Timer.SECOND)
    timer.start()
    with mp.Pool(processes=get_num_processes(flags)) as pool:
        if not flags.viz:
            pool.apply_async(start_listen, (q, tmp_label_path))

        pool.starmap(generate_img, zip(
            range(start_index, start_index + flags.num_img), repeat(q)))

        q.put(STOP_TOKEN)
        pool.close()
        pool.join()
    timer.end("Finish generate data")

    if not flags.viz:
        sort_labels(tmp_label_path, label_path)

import h5py
import numpy as np


def load_model(fn, key='data'):
    fid = h5py.File(fn, 'r')
    data = np.array(fid.get(key))
    fid.close()
    return data


def load_content_model(fn, key='data'):
    fid = h5py.File(fn, 'r')
    data = fid[key]
    return data


def save_model(fn, data):
    fid = h5py.File(fn, 'w')
    fid.create_dataset('data', data=data)
    fid.close()
    return True


def parse_content_style_size(args):
    # Parse content_size / style_size
    # convert to [x,y]
    def parse_size(input):
        if isinstance(input, str):
            # String 'x,y'
            input_list = list(map(lambda x: int(x), input.split(',')))
        elif isinstance(input, int):
            # int x
            input_list = [input]
        else:
            # List [x,y]
            input_list = input
        if len(input_list) == 1:
            input_list *= 3
        return input_list
    if args.content_size is not None:
        args.content_size = parse_size(args.content_size)
    if args.style_size is not None:
        args.style_size = parse_size(args.style_size)
    return args


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def gram_matrix(y):
    (b, ch, d, h, w) = y.size()
    features = y.view(b, ch, d*h*w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w * d)
    return gram


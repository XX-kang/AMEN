"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np
import torch.nn as nn

from collections import OrderedDict

def make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.root.setLevel(logging.INFO)
    return logger


def trans_dict_to_xml(data):
    xml = []
    for k in sorted(data.keys()):
        v = data.get(k)
        if k == 'detail' and not v.startswith('<![CDATA['):
            v = '<![CDATA[{}]]>'.format(v)
        xml.append('<{key}>{value}</{key}>'.format(key=k, value=v))
    return '<xml>{}</xml>'.format(''.join(xml))


def generate_yaml_doc_ruamel(data, yaml_file):
    from ruamel import yaml

    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(data, file, Dumper=yaml.RoundTripDumper)
    file.close()


def get_yaml_data_ruamel(yaml_file):
    from ruamel import yaml
    file = open(yaml_file, 'r', encoding='utf-8')
    data = yaml.load(file.read(), Loader=yaml.Loader)
    file.close()
    return data


def get_args_from_dict(data):
    # data is a dict object
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--estimators', nargs="+", required=True,
                        choices=data)
    args = vars(parser.parse_args())

    return args


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)

def append_yaml_doc_ruamel(data, yaml_file):
    from ruamel import yaml

    file = open(yaml_file, 'a', encoding='utf-8')
    yaml.dump(data, file, Dumper=yaml.RoundTripDumper)
    file.close()

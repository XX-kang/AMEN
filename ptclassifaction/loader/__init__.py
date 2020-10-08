import json

from ptclassifaction.loader.imagefolder_loader import ImageFolderLoader
from ptclassifaction.loader.h5py_loader import H5pyLoader
from ptclassifaction.loader.cla_seg_loader import ClaSegLoader
from ptclassifaction.loader.deeplesion_loader import DeepLesionLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "imagefolder": ImageFolderLoader,
        "h5py": H5pyLoader,
        'clasegloader': ClaSegLoader,
        'deeplesionloader': DeepLesionLoader,
    }[name]

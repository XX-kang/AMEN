import copy
import torchvision.models as models

from ptclassifaction.models.vgg_cam import vgg16_cam, vgg19_cam
from ptclassifaction.models.vgg import vgg16, vgg19
from ptclassifaction.models.resnet import resnet50,resnet101
from ptclassifaction.models.squeezenet import squeezenet1_0, squeezenet1_1
from ptclassifaction.models.inception import inception_v3
from ptclassifaction.models.densenet import densenet121, densenet161, densenet169, densenet201
from ptclassifaction.models.googlenet import googlenet
from ptclassifaction.models.mobilenetv1 import MobileNet as mobilenetv1
from ptclassifaction.models.mobilenetv2 import MobileNetV2 as mobilenetv2


def get_model(model_dict, pretrained, num_classes, input_channels, version=None):
    name = model_dict
    model = _get_model_instance(name)
    param_dict = {}

    if name in ["vgg16", "vgg19"]:
        model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)

    if name in ["vgg16_cam", "vgg19_cam"]:
        model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)

    elif name in ["resnet50", "resnet101"]:
        model = model(pretrained=pretrained, num_classes=num_classes, input_channels=input_channels, **param_dict)

    elif name in ["squeezenet1_0", "squeezenet1_1"]:
        model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)

    elif name == "mobilenetv1":
        model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)

    elif name == "mobilenetv2":
        model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)

    elif name == "inception_v3":
        model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)

    elif name in ["densenet121", "densenet161", "densenet169", "densenet201"]:
        model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)

    else: #googlenet
         model = model(num_classes=num_classes, input_channels=input_channels, **param_dict)


    return model

def _get_model_instance(name):
    try:
        return {
            "vgg16": vgg16,
            "vgg19": vgg19,
            "vgg16_cam": vgg16_cam,
            "vgg19_cam": vgg19_cam,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "squeezenet1_0": squeezenet1_0,
            "squeezenet1_1": squeezenet1_1,
            "inception_v3": inception_v3,
            "densenet121": densenet121,
            "densenet161": densenet161,
            "densenet169": densenet169,
            "densenet201": densenet201,
            "googlenet": googlenet,
            "mobilenetv1":mobilenetv1,
            "mobilenetv2": mobilenetv2,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def get_finalconv_name(name):
    try:
        return {
            # "vgg16": vgg16,
            # "vgg19": vgg19,
            "resnet50": "layer4",
            "resnet101": "layer4",
            "squeezenet1_0": 'features',
            "squeezenet1_1": 'Conv2d-63',   #[-1, 2, 15, 15]
            "inception_v3": 'Mixed_7c',
            "densenet121": 'features',
            "densenet161": 'features',
            "densenet169": 'features',
            "densenet201": 'features',
            "googlenet": 'inception5b',
        }[name]
    except:
        raise ("Model {} final conv not available".format(name))
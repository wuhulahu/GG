
from models import resnet50, googlenetv3
from timm.models import create_model
# from models.crossvit_cut_token import crossvit_small_224_cut
from models.crossvit_cut_token_debug import crossvit_small_224_cut_debug
from models.TransformerModel.modeling import VisionTransformer_load
from models.ViTHashing import vit_load

def load_model(args, code_length, erasing_model_path=None, **kwargs):
    """
    Load cnn model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.
        erasing_model_path(str): evaluate_erasing_model_path

    Returns
        model(torch.nn.Module): CNN model.
    """
    arch = args.arch
    code_length = code_length
    pretrain = args.pretrain

    synchronized = args.synchronized    # 控制msvit两个通道的mask比例是否相似
    soft_per_example = args.soft_per_example  # 是一个batch一个soft，还是对每个样例进行soft
    weight_remain = args.weight_remain
    # layer_cut = args.layer_cut  # 从第几层开始cut
    sigma = args.sigma
    limit = args.limit

    if arch == 'resnet50':
        model = resnet50.load_model(code_length, args.weight_remain, args.cut, args.sigma, args.limit, use_custom_activation=args.use_custom_activation, ratio_att=args.ratio_att, erasing_model_path=erasing_model_path)
    elif arch == 'googlenetv3':
        model = googlenetv3.load_model(code_length, args.weight_remain, args.cut, args.sigma, args.limit, use_custom_activation=args.use_custom_activation, ratio_att=args.ratio_att, erasing_model_path=erasing_model_path)

    elif 'ViTHashing' in arch:
        model = vit_load(arch=arch, code_length=code_length, weight_remain=weight_remain, sigma=sigma, limit=limit,
                         cut=args.cut,use_custom_activation=args.use_custom_activation, ratio_att=args.ratio_att,
                         erasing_model_path=erasing_model_path)

    elif 'cut' not in arch:
        # 也就是crossvit
        model = create_model(
            arch,
            pretrained=pretrain,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.1,
            code_length=code_length,
            drop_block_rate=None,
        )
    else:
        # crossvit_cut_token_debug
        model = create_model(
            arch,
            pretrained=pretrain,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.1,
            code_length=code_length,
            synchronized=synchronized,
            weight_remain=weight_remain,
            soft_per_example=soft_per_example,
            cut=args.cut,
            sigma=sigma,
            limit=limit,
            drop_block_rate=None,
            use_custom_activation=args.use_custom_activation,
            ratio_att=args.ratio_att,
            erasing_model_path=erasing_model_path
        )


    return model

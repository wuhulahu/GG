import data.cifar10 as cifar10
import data.nus_wide as nuswide
import data.imagenet as imagenet
import data.flickr25k as flickr
import data.coco as coco
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(args, only_query=False, toy_test_dismatch=False):

    dataset = args.dataset
    root = args.root
    num_query = args.num_query
    num_train = args.num_train
    batch_size = args.batch_size
    img_size = args.img_size
    num_workers = args.num_workers
    sigma_crop = args.sigma_crop
    soft = args.soft
    wipe = args.wipe
    erasing = args.erasing
    compare_flag = args.compare_flag
    if dataset == 'cifar-10':
        train_dataloader, query_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     img_size,
                                                                                     num_workers,
                                                                                     num_class=10,
                                                                                     soft=soft,
                                                                                     sigma_crop=sigma_crop,
                                                                                     wipe=wipe,
                                                                                     sigma=args.sigma,
                                                                                     erasing=erasing,
                                                                                     only_query=only_query,
                                                                                     compare_flag=compare_flag
                                                                                     )
    elif dataset == 'nus-wide-tc21':
        train_dataloader, query_dataloader, retrieval_dataloader = nuswide.load_data(21,
                                                                                     root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     img_size,
                                                                                     num_workers,
                                                                                     num_class=21,
                                                                                     soft=soft,
                                                                                     sigma_crop=sigma_crop,
                                                                                     wipe=wipe,
                                                                                     sigma=args.sigma,
                                                                                     erasing=erasing,
                                                                                     compare_flag=compare_flag
                                                                                     )
    elif dataset == 'imagenet':
        train_dataloader, query_dataloader, retrieval_dataloader = imagenet.load_data(root, 
                                                                                      batch_size,
                                                                                      img_size,
                                                                                      num_workers,
                                                                                      num_class=100,
                                                                                      soft=soft,
                                                                                      sigma_crop=sigma_crop,
                                                                                      wipe=wipe,
                                                                                      sigma=args.sigma,
                                                                                      erasing=erasing,
                                                                                      compare_flag=compare_flag
                                                                                      )

    elif dataset == 'coco':
        train_dataloader, query_dataloader, retrieval_dataloader = coco.load_data(root,
                                                                                  batch_size,
                                                                                  img_size,
                                                                                  num_workers,
                                                                                  num_class=38,
                                                                                  soft=soft,
                                                                                  sigma_crop=sigma_crop,
                                                                                  wipe=wipe,
                                                                                  sigma=args.sigma,
                                                                                  erasing=erasing,
                                                                                  compare_flag=compare_flag
                                                                                  )
    else:
        raise ValueError("Invalid dataset name!")

    return train_dataloader, query_dataloader, retrieval_dataloader


def load_data_visual(dataset, root, num_query, num_train, toy_test_dismatch=False):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        query_dataset, retrieval_dataset = cifar10.load_data_visual(root,
                                                                    num_query,
                                                                    num_train,
                                                                    toy_test_dismatch=toy_test_dismatch
                                                                    )
    elif dataset == 'imagenet':
        query_dataset, retrieval_dataset = imagenet.load_data_visual(root)

    else:
        raise ValueError("Invalid dataset name!")

    return query_dataset, retrieval_dataset



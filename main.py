import os
import sys, rich, torch, time, random
import numpy as np

from Tasks.Trainer import Trainer
from Dataloaders import *
from Data.preprocess import load_amazon_data
from Utils.configs import DomainAdaptation, SourceOnly, SignleSourceDomainAdaptation
from Utils.logger import get_rich_logger


def main(task: str="DA"):
    if task == "MSUDA":
        config = DomainAdaptation.parse_arguments()
    elif task == "Source":
        config = SourceOnly.parse_arguments()
    elif task == "UDA": # single source
        config = SignleSourceDomainAdaptation.parse_arguments()
    else:
        NotImplementedError

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in config.gpus])

    # Seed Fix
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    """Configs Printed and Saved"""
    rich.print(config.__dict__)
    ckpt_dir = config.checkpoint_dir + "_seed" + str(config.seed)
    config.save(ckpt_dir)

    rich.print(f"Training Start")
    main_worker(0, ckpt_dir, config=config)


def main_worker(local_rank:int, ckpt_dir, config:object):

    # Set default gpus number
    torch.cuda.set_device(local_rank)

    # Prepare
    domain_names = load_amazon_data(config.data, config.data_dir)[0]

    # Checkpoint dir
    os.makedirs(ckpt_dir, exist_ok=True)
    logfile = os.path.join(ckpt_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    logger.info(f'Checkpoint directory: {ckpt_dir}')

    # Model Train & Evaluate 
    start = time.time()
    kwargs = {'epochs': config.epochs, 'epochs_step1': config.epochs_step1, 'epochs_step2': config.epochs_step2,
              'epochs_step3' : config.epochs_step3, 'epochs_alpha': config.epochs_alpha, 'batch_size': config.batch_size, 'unc_type':config.unc_type}
    model = Trainer(config, local_rank, config.seed, ckpt_dir, domain_names, **kwargs)

    model.run(logger=logger)
    model.save_testing_result()

    end_sec = time.time() - start

    if logger is not None:
        end_min = end_sec / 60
        logger.info(f"Total Training Time: {end_min: 2f} minutes")
        logger.handlers.clear()


if __name__ == '__main__':
    try:
        """["MSUDA", "Source", "UDA"]"""
        main(task="MSUDA")

    except KeyboardInterrupt:
        sys.exit(0)
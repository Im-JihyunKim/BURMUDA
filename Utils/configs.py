import os, json, copy, datetime, argparse


class ConfigBase(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        parents = [
            cls.data_parser(),
            cls.model_parser(),
            cls.train_parser(),
            cls.logging_parser(),
            cls.task_specific_parser()
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='0')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)

        return config

    @classmethod
    def form_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, ckpt_dir):
        path = os.path.join(ckpt_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        raise NotImplementedError

    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.task,
            self.data,
            self.hash
        )
        return ckpt

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data-dir', type=str, default='./Data/')
        parser.add_argument('--data', type=str, default='amazon1', choices=('amazon1', 'amazon2', 'amazon3'))
        parser.add_argument('--seed', type=int, default=418)

        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("BURMUDA Model", add_help=False)
        parser.add_argument('--backbone-type', type=str, default='MLP', choices=('MLP', 'LSTM', 'BERT'))
        parser.add_argument('--input-dim', type=int, default=5000, choices=(5000, 768),
                            help='5000 for amazon1, 768 for amazon3')
        parser.add_argument('--num-classes', type=int, default=2, choices=(2, 5),
                            help='Binary classification')

        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size.')
        parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate to start from.')
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight-decay", default=0., type=float, help='l2 weight decay')
        parser.add_argument("--clip", default=1, type=int, help="Norm of Gradient Clipping")
        parser.add_argument('--optimizer', type=str, default='adam', choices=('sgd', 'adam', 'adamw', 'lookahead'),
                            help='Optimization algorithm.')
        parser.add_argument('--lr-scheduler', default='cosine',
                            choices=(None, 'exponential', 'step', 'cosine', 'restart'))
        parser.add_argument('--gpus', type=str, nargs='+', default='0', help='')
        parser.add_argument('--num_dropout', type=int, default=30)
        parser.add_argument('--dropout_mode', type=str, default='Combine',
                            choices=('Combine', 'Feature_extractor_only', 'Classifier_only', 'No'))
        parser.add_argument('--unc_scale', type=float, default=1e-5)
        parser.add_argument('--unc_type', default='logit', choices=('logit', 'prob'))
        parser.add_argument('--unc_calculation_type', default='minimize', type=str, choices=('minimize', 'adversarial'))

        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint-root', type=str, default='./Results/',
                            help='Top-level directory of checkpoints.')
        return parser


# Task specific parser
class DomainAdaptation(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(DomainAdaptation, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('', add_help=False)
        parser.add_argument('--epochs', default=50, type=int, help="epochs for adaptation")
        parser.add_argument('--pretraining', default=True, action='store_true')
        parser.add_argument('--short_pretraining', default=True, type=bool)
        parser.add_argument('--epochs-pretrain', default=10, type=int)
        parser.add_argument('--epochs-step1', default=1, type=int)
        parser.add_argument('--epochs-step2', default=1, type=int)
        parser.add_argument('--epochs-step3', default=2, type=int)
        parser.add_argument('--epochs-alpha', default=1, type=int)
        parser.add_argument('--lam-alpha', default=1, type=float)
        parser.add_argument('--lr-alpha', default=1e-4, type=float)
        parser.add_argument('--mu', default=0, type=float)
        parser.add_argument('--pretrained_datetime', default='2023-05-29_16-08-13', type=str, help='datetime for Source_Only Model Training')
        parser.add_argument('--pretrained_epoch', default=10, type=int)

        return parser

    @property
    def task(self) -> str:
        return "Domain_Adaptation"


class SourceOnly(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SourceOnly, self).__init__(args, *kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('', add_help=False)
        parser.add_argument("--epochs", default=50, type=int, help="epochs for supervised learning with source domain")
        parser.add_argument('--epochs-step1', default=1, type=int)
        parser.add_argument('--epochs-step2', default=0, type=int)
        parser.add_argument('--epochs-step3', default=0, type=int)
        parser.add_argument('--epochs-alpha', default=0, type=int)
        parser.add_argument('--lr-alpha', default=5e-4, type=float)
        parser.add_argument('--save_epoch', default=[10, 30, 50, 70, 100], type=list)
        
        return parser

    @property
    def task(self) -> str:
        return "Source_Only"


class SignleSourceDomainAdaptation(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(SignleSourceDomainAdaptation, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('', add_help=False)
        parser.add_argument('--pretraining', default=True, action='store_true')
        parser.add_argument('--epochs', default=100, type=int, help="epochs for adaptation")
        parser.add_argument('--epochs-step1', default=1, type=int)
        parser.add_argument('--epochs-step2', default=1, type=int)
        parser.add_argument('--epochs-step3', default=1, type=int)
        parser.add_argument('--epochs-alpha', default=0, type=int)
        parser.add_argument('--lam-alpha', default=0.01, type=float)
        parser.add_argument('--pretrained_datetime', default='2023-05-29_16-08-13', type=str, help='datetime for Source_Only Model Training')
        parser.add_argument('--pretrained_epoch', default=10, type=int)
        parser.add_argument('--lr-alpha', default=1e-3, type=float)
        parser.add_argument('--mu', default=0, type=float)

        return parser

    @property
    def task(self) -> str:
        return "Single_Source_Adaptation"
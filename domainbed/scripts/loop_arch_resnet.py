import os
import time
import argparse

import warnings

warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Domain generalization")

parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--mix_strategy", type=str, default=None)

parser.add_argument("--dataset_group", type=str, default='A')
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--algorithm", type=str, default='ERM')
parser.add_argument("--architecture", type=str, default='resnet50', help='alternatives: resnet50_nopretraining, vit-b')
parser.add_argument("--da_bool", type=str2bool, default=False, help='true for data augmentation')
parser.add_argument("--da_strategy", type=str, default='no-no', help="""
                    format for this argument is: random_shuffle-Mixup, random_shuffle-CutMix, LISA-Mixup, LISA-CutMix
                    """)
parser.add_argument("--n_iter", type=int, default=3)

args = parser.parse_args()

data_dir = "./data/spawrious224"
batch_size = 128
# batch_size = 2


#     elif algorithm == "W2D":
#         _hparam('rsc_f_drop_factor', 1 / 4, lambda r: r.uniform(0.1, 0.4))
#         _hparam('last_k_epoch', 1 / 4, lambda r: r.uniform(0.2, 0.4))
#         if dataset in SMALL_DATASET:
#             _hparam('rsc_b_drop_factor', 1 / 4, lambda r: r.uniform(0.1, 0.3))
#             _hparam('worst_case_p', 1 / 3, lambda r: r.uniform(0.1, 0.5))
#         else:
#             _hparam('rsc_b_drop_factor', 1 / 3, lambda r: r.uniform(0.1, 0.4))
#             _hparam('worst_case_p', 1 / 3, lambda r: r.uniform(0.2, 0.4))

hparams_dict = {
    "SpawriousO2O_easy": {
        "ERM": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06}""",
        "W2D": """{
                "batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "rsc_f_drop_factor": 1 / 4,
                "last_k_epoch": 1 / 4,
                "rsc_b_drop_factor": 1 / 3,
                "worst_case_p": 1 / 3}""",
        "JTT": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06}""",
        "LLR": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06}""",
        "FLR": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06}""",
        "GroupDRO": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "groupdro_eta": 0.0053050580120662895} """,
        "IRM": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "irm_lambda": 1.8838285530562104,
                "irm_penalty_anneal_iters": 247} """,
        "MMD": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "mmd_gamma": 7.289784897124338} """,
        "CORAL": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "mmd_gamma": 6.9018246989615895} """,
        "CausIRL_CORAL": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "mmd_gamma": 3.5146823420446407} """,
        "Fish": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "meta_lr": 0.5} """,
        "VREx": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "vrex_penalty_anneal_iters": 8,
                "vrex_lambda": 0.14959251216362196} """,
        "DANN": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06},
                "d_steps_per_g_step": 3,
                "grad_penalty": 2.7663277782638813,
                "lambda": 51.30168523785897,
                "lr_d": 7.400983580823492e-05,
                "lr_g": 3.933474398919139e-05,
                "mlp_depth": 3,
                "mlp_dropout": 0.1,
                "mlp_width": 86,
                "weight_decay_d": 2.988916639814755e-05,
                "weight_decay_g": 1.2253150874974055e-05,
                "beta1": 0.0}""",
        "CAD": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06
                "temperature": 0.1}""",
    },
    "SpawriousM2M_hard": {
        "ERM": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05}""",
        "W2D": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05
                "rsc_f_drop_factor": 1 / 4,
                "last_k_epoch": 1 / 4,
                "rsc_b_drop_factor": 1 / 3,
                "worst_case_p": 1 / 3}""",
        "JTT": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05}""",
        "LLR": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05}""",
        "FLR": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05}""",
        "GroupDRO": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "groupdro_eta": 0.013378423587817576} """,
        "IRM": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "irm_lambda": 29.3676220201571,
                "irm_penalty_anneal_iters": 3001} """,
        "MMD": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "mmd_gamma": 1.0215072228839979} """,
        "CORAL": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "mmd_gamma": 0.5870292457165399} """,
        "CausIRL_CORAL": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "mmd_gamma": 0.5870292457165399} """,
        "Fish": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "meta_lr": 0.5} """,
        "VREx": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "vrex_penalty_anneal_iters": 8,
                "vrex_lambda": 0.14959251216362196} """,
        "DANN": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06},
                "d_steps_per_g_step": 1,
                "grad_penalty": 3.044417571888086,
                "lambda": 0.04765466589023375,
                "lr_d": 0.00012638696055528178,
                "lr_g": 2.6800465071227878e-05,
                "mlp_depth": 4,
                "mlp_dropout": 0.1,
                "mlp_width": 509,
                "weight_decay_d": 0.006008142228932282,
                "weight_decay_g": 6.355656257759041e-05,
                "beta1": 0.0}""",
        "CAD": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06
                "temperature": 0.1}""",
    },
}

hparams_dict["SpawriousO2O_medium"] = hparams_dict["SpawriousO2O_easy"]
hparams_dict["SpawriousO2O_hard"] = hparams_dict["SpawriousO2O_easy"]

hparams_dict["SpawriousM2M_easy"] = hparams_dict["SpawriousM2M_hard"]
hparams_dict["SpawriousM2M_medium"] = hparams_dict["SpawriousM2M_hard"]

dataset_group = args.dataset_group
algo = args.algorithm
arch = args.architecture
da_strategy = args.da_strategy
da_bool = args.da_bool
dataset = args.dataset

mix_strategy = da_strategy.split('-')[0]
mix_interpolation = da_strategy.split('-')[1]

dataset_group_dict = {
        "A": ["SpawriousO2O_easy", "SpawriousO2O_medium"],
        "B": ["SpawriousO2O_hard", "SpawriousM2M_easy"],
        "C": ["SpawriousM2M_medium", "SpawriousM2M_hard"],
}

every_dataset_group_dict = {
        "A": ["SpawriousO2O_easy"],
        "B": ["SpawriousO2O_medium"],
        "C": ["SpawriousO2O_hard"],
        "D": ["SpawriousM2M_easy"],
        "E": ["SpawriousM2M_medium"],
        "F": ["SpawriousM2M_hard"],
}

count = 0

if dataset is None:
        if arch == "vit-b":
                dataset_list = every_dataset_group_dict[dataset_group]
        else:
                dataset_list = dataset_group_dict[dataset_group]
else:
        dataset_list = [dataset]

# TODO: remove this temp code
dataset_list = ["SpawriousM2M_hard"]

# sleep for 4 hours 

arch = 'vit-foundation'
# arch = 'vit-b'
# for dataset in dataset_group_dict[dataset_group]:
for dataset in dataset_list:

        count += 1
        print(f"\n\n\nCount: {count}\n\n\n")

        hparams = (
        hparams_dict[dataset]['ERM']
        .replace("batchsize", str(batch_size))
        .replace("archused", arch)
        )
        hparams = hparams.replace("\n", "").replace(" ", "")
        # algo = "CutMix"

        print(f"Train {algo} on {dataset}")

        if da_bool:
                print('about to sleep for 4 hours')
                # time.sleep(14400)
                print('done sleeping')
                os.system(
                f"""python3 -m domainbed.scripts.train_n --data_dir={data_dir}  --algorithm {algo} --test_env 0 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir mixup2-rebuttal-results --n_iter 1 --mix_strategy {mix_strategy} --mix_interpolation {mix_interpolation}"""
                )
                # os.system(
                # f"""python3 -m domainbed.scripts.train_n --data_dir={data_dir}  --algorithm ERM --test_env 0 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir erm-rebuttal-results --n_iter 1"""
                # )
                
        else:
                
                os.system(f"""python3 -m domainbed.scripts.train_n --data_dir={data_dir}  --algorithm {algo} --test_env 0 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir {algo}-{arch}-{da_strategy}-results --n_iter {args.n_iter} """)
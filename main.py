import argparse
import os
import math
import torch
import json
import copy
import random
import numpy as np
from tqdm import tqdm
from dataset import dataset_constructor
from model import model_selector, get_df_parameter, model_train, model_test


def get_parameter(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', default='chembl_bdb', type=str)
    parser.add_argument('--model_name', default='model', type=str)
    parser.add_argument('--test_ref_num', default=0.3, type=str)
    parser.add_argument('--test_repeat_num', default=1, type=int)

    parser.add_argument('--logdir', default="./checkpoints/checkpoints_chembl_bdb", type=str, help='directory for summaries and checkpoints.')
    parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    df_group = parser.add_argument_group('Default Parameters')
    param_group = parser.add_argument_group('Main Parameters')
    get_df_parameter(df_group)
    get_parameter(param_group)

    args = parser.parse_args()
    try:
        args.test_ref_num = json.loads(args.test_ref_num)
        print('args.test_ref_num:', args.test_ref_num)
    except:
        args.test_ref_num = float(args.test_ref_num)
        if args.test_ref_num > 1:
            args.test_ref_num = int(args.test_ref_num)
        print('args.test_ref_num:', args.test_ref_num)

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    random.seed(1)
    np.random.seed(2)

    datasource = args.datasource
    exp_string = f'data_{datasource}_{args.model_name}'

    model = model_selector(args)(args=args, input_shape=(2, args.dim_w))
    dataloader = dataset_constructor(args)

    if args.train == 1:
        if args.resume == 1:
            model_file = 'D:/Author/code/model/checkpoints/chembl_bdb/model_best'
            print("resume training from", model_file)
            try:
                model.load_state_dict(torch.load(model_file))
            except Exception as e:
                model.load_state_dict(torch.load(model_file), strict=False)
        model_train(args, model, dataloader)
    elif args.train == 0:
        args.meta_batch_size = 1
        model_file = 'D:/Author/code/model/checkpoints/chembl_bdb/model_best'

        if not isinstance(args.test_ref_num, list):
            args.test_ref_num = [args.test_ref_num]
        test_ref_num_all = copy.deepcopy(args.test_ref_num)
        for test_ref_num in test_ref_num_all:
            args.test_ref_num = test_ref_num
            try:
                model.load_state_dict(torch.load(model_file))
            except Exception as e:
                print(e)
                model.load_state_dict(torch.load(model_file), strict=False)
            res_dict, _ = model_test(args, args.test_epoch, model, dataloader)

    elif args.train == 2:
        test_data_all = dataloader.get_test_batches()
        model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
        if not os.path.exists(model_file):
            model_file = '{0}/{1}/model_best'.format(args.logdir, exp_string)
        try:
            model.load_state_dict(torch.load(model_file))
        except Exception as e:
            print(e)
            model.load_state_dict(torch.load(model_file), strict=False)

        exit()
# train command
'''
python main_reg.py --train 1 --datasource chembl --model_name actfound

python main_reg.py --train 1 --datasource chembl_bdb --model_name actfound --test_sup_num 0.3

FIXED_PARAM="--test_sup_num 16 --test_repeat_num 1 --train 1 --expert_test("") --begin_lrloss_epoch 50 --metatrain_iterations 80 --no_fep_lig "
'''
# test command
'''
FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test act_cliff"

python main_reg.py --datasource gdsc --test_write_file test_results/gdsc --test_repeat_num 1 --metatrain_iterations 85 --input_celline --resume 1 --test_epoch -1 --gdsc_pretrain chembl

python main_reg.py --datasource gdsc --test_write_file test_results/gdsc --test_repeat_num 1 --metatrain_iterations 85 --input_celline --resume 1 --test_epoch -1 --gdsc_pretrain bdb

python main_reg.py --datasource gdsc --test_write_file test_results/gdsc --test_repeat_num 1 --metatrain_iterations 85 --resume 1 --test_epoch -1 --gdsc_pretrain chembl

python main_reg.py --datasource gdsc --test_write_file test_results/gdsc --test_repeat_num 1 --metatrain_iterations 85 --resume 1 --test_epoch -1 --gdsc_pretrain bdb

--logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES_GDSC="./test_results/gdsc"
python main_reg.py --datasource=chembl --test_sup_num 0.5 --test_repeat_num 1 --train 0 --test_epoch -1 --expert_test gdsc --model_name actfound
python main_reg.py --datasource=bdb --test_sup_num 0.5 --test_repeat_num 1 --train 0 --test_epoch -1 --expert_test gdsc --model_name actfound

FIXED_PARAM_DAVIS="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test davis"
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES_DAVIS="./test_results/result_cross/chembl2davis"
python main_reg.py --datasource=chembl  --model_name actfound --test_sup_num 0.3 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test davis
python main_reg.py --datasource=chembl  --model_name actfound --test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test davis

python main_reg.py --datasource=chembl  --model_name actfound --test_sup_num 0.3 --test_repeat_num 2 --train 0 --test_epoch -1 --expert_test act_cliff

python main_reg.py --train 1 --datasource chembl --model_name camp --test_sup_num 0.2
python main_reg.py --train 1 --datasource bdb --model_name actfound --test_sup_num 0.1

python main_reg.py --train 1 --datasource chembl_bdb --model_name camp --test_sup_num 0.3

python main_reg.py --datasource=chembl_bdb  --model_name protonet --test_sup_num 0.3 --test_repeat_num 1 --train 0 --test_epoch -1 --expert_test act_cliff
python main_reg.py --datasource=chembl_bdb  --model_name camp --test_sup_num 8 --test_repeat_num 1 --train 0 --test_epoch -1 --expert_test act_cliff

python main_reg.py --train 1 --resume 1 --datasource chembl_bdb --model_name camp --test_sup_num 0.3

python main_reg.py --datasource=chembl_bdb  --model_name camp --test_sup_num 1 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test davis
python main_reg.py --datasource gdsc_davis --test_write_file test_results/gdsc --test_sup_num 0 --test_repeat_num 1 --metatrain_iterations 85 --resume 1 --test_epoch -1 --gdsc_pretrain bdb

python main_reg.py --datasource=chembl_bdb  --model_name camp --test_sup_num 32 --test_repeat_num 2 --train 0 --test_epoch -1 --expert_test act_cliff

python main.py --datasource=chembl_bdb  --model_name model --test_ref_num 0.3
'''

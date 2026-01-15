import argparse
import torch
import numpy as np
from model.model import ModelRegressor

def model_selector(args):
    model_name = args.model_name
    if model_name == "model":
        return ModelRegressor

    raise ValueError(f"model {model_name} is not supported")

def get_df_parameter(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--dim_w', default=2048, type=int, help='dimension of w')
    parser.add_argument('--hid_dim', default=2048, type=int, help='dimension of w')
    parser.add_argument('--num_stages', default=2, type=int, help='num stages')
    parser.add_argument('--per_step_bn_statistics', default=True, action='store_false')
    parser.add_argument('--learnable_bn_gamma', default=True, action='store_false', help='learnable_bn_gamma')
    parser.add_argument('--learnable_bn_beta', default=True, action='store_false', help='learnable_bn_beta')
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', default=False, action='store_true', help='enable_inner_loop_optimizable_bn_params')
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', default=True, action='store_false', help='learnable_per_layer_per_step_inner_loop_learning_rate')
    parser.add_argument('--use_multi_step_loss_optimization', default=True, action='store_false', help='use_multi_step_loss_optimization')
    parser.add_argument('--second_order', default=1, type=int, help='second_order')
    parser.add_argument('--first_order_to_second_order_epoch', default=10, type=int, help='first_order_to_second_order_epoch')

    parser.add_argument('--expert_test', default="", type=str)
    parser.add_argument('--similarity_cut', default=1., type=float)

    parser.add_argument('--train_seed', default=1111, type=int, help='train_seed')
    parser.add_argument('--val_seed', default=1111, type=int, help='val_seed')
    parser.add_argument('--test_seed', default=1111, type=int, help='test_seed')

    parser.add_argument('--metatrain_iterations', default=80, type=int,
                        help='number of metatraining iterations.')
    parser.add_argument('--meta_batch_size', default=16, type=int, help='number of tasks sampled per meta-update')
    parser.add_argument('--min_learning_rate', default=0.0001, type=float, help='min_learning_rate')
    parser.add_argument('--update_lr', default=0.001, type=float, help='inner learning rate')
    parser.add_argument('--meta_lr', default=0.00015, type=float, help='the base learning rate of the generator')
    parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--test_num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--multi_step_loss_num_epochs', default=5, type=int, help='multi_step_loss_num_epochs')
    parser.add_argument('--norm_layer', default='batch_norm', type=str, help='norm_layer')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in beta distribution')

    parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
    parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')
    parser.add_argument('--use_byhand_lr', default=False, action='store_true')

    return parser

def model_train(args, model, dataloader):
    exp_string = f'data_{args.datasource}_{args.model_name}'
    Print_Iter = 200

    begin_epoch = 0
    if args.resume == 1:
        begin_epoch = args.test_epoch + 1

    _, last_test_result = model_test(args, begin_epoch, model, dataloader, is_test=False)

    best_epoch = -1
    print_loss = 0.0
    print_step = 0
    for epoch in range(begin_epoch, args.metatrain_iterations):
        train_data_all = dataloader.get_train_batches()
        for step, cur_data in enumerate(train_data_all):
            meta_batch_loss, _ = model.run_train_iter(cur_data, epoch)

            if (print_step+1) % Print_Iter == 0 or step == len(train_data_all)-1:
                print('epoch: {}, iter: {}, mse: {}'.format(epoch, step, print_loss/print_step))
                print_loss = 0.0
                print_step = 0
            else:
                print_loss += meta_batch_loss['loss']
                print_step += 1

        _, test_result = model_test(args, epoch, model, dataloader, is_test=False)
        torch.save(model.state_dict(), '{0}/{2}/model_{1}'.format(args.logdir, epoch, exp_string))
        if last_test_result < test_result:
            last_test_result = test_result
            best_epoch = epoch
            torch.save(model.state_dict(), '{0}/{2}/model_best'.format(args.logdir, epoch, exp_string))
        print("best valid epoch is:", best_epoch)


def model_test(args, epoch, model, dataloader, is_test=True):

    cir_num = args.test_repeat_num
    r2_list = []
    rmse_list = []
    res_dict = {}
    # print(f"cir_num:{cir_num}")
    for cir in range(cir_num):
        if is_test:
            test_data_all = dataloader.get_test_batches(repeat_cnt=cir)
        else:
            test_data_all = dataloader.get_val_batches(repeat_cnt=cir)
        for step, cur_data in enumerate(test_data_all):
            ligands_x = cur_data[0][0]
            if len(ligands_x) <= args.test_ref_num:
                continue
            losses, per_task_target_preds, final_weights, per_task_metrics = model.run_validation_iter(cur_data)
            r2_list.append(per_task_metrics[0]["r2"])
            rmse_list.append(per_task_metrics[0]["rmse"])
            target_name = cur_data[3][0]
            if target_name not in res_dict.keys():
                res_dict[target_name] = []
            res_dict[target_name].append(per_task_metrics[0])

    rmse_i = np.mean(rmse_list)
    median_r2 = np.median(r2_list, 0)
    mean_r2 = np.mean(r2_list, 0)
    valid_cnt = len([x for x in r2_list if x > 0.3])
    print(
        'epoch is: {}, mean rmse is: {:.3f}'.
        format(epoch, rmse_i))
    print(
        'epoch is: {}, r2: mean is: {:.3f}, median is: {:.3f}, cnt>0.3 is: {:.3f}'.
        format(epoch, mean_r2, median_r2, valid_cnt))
    return res_dict, mean_r2-rmse_i+1

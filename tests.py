import unittest
import numpy as np
import os
import torch
import shutil

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jobs as hjb


class Test(unittest.TestCase):

    def test_cartesian_product(self):
        # test whether the cartesian product covers all needed variations
        exp_dict_1 = {'dataset': 'mnist', 'model': 'mlp', 'batch_size': 1}
        exp_dict_2 = {'dataset': 'mnist', 'model': 'mlp', 'batch_size': 5}
        exp_dict_3 = {'dataset': 'cifar10', 'model': 'mlp', 'batch_size': 1}
        exp_dict_4 = {'dataset': 'cifar10', 'model': 'mlp', 'batch_size': 5}
        exp_list = [exp_dict_1, exp_dict_2, exp_dict_3, exp_dict_4]
        exp_list_cartesian = hu.cartesian_exp_group(
            {'dataset': ['mnist', 'cifar10'], 'model': 'mlp',
             'batch_size': [1, 5]})
        exp_list_hash = [hu.hash_dict(e) for e in exp_list]
        exp_list_cartesian_hash = [hu.hash_dict(e) for e in exp_list_cartesian]

        # check if the # experiments is correct
        assert(len(exp_list_cartesian_hash) == len(exp_list_hash))

        # check that the hashes in the cartesian are all there
        for h in exp_list_hash:
            assert(h in exp_list_cartesian_hash)

        # check that every hash is unique
        assert (len(exp_list_cartesian_hash) ==
                len(np.unique(exp_list_cartesian_hash)))

    def test_hash(self):
        # test whether hashing works for nested dicts
        exp_dict_1 = {'model': {'name': 'mlp', 'n_layers': 30},
                      'dataset': 'mnist', 'batch_size': 1}
        exp_dict_2 = {'dataset': 'mnist', 'batch_size': 1,
                      'model': {'name': 'mlp', 'n_layers': 30}}
        exp_dict_3 = {'dataset': 'mnist', 'batch_size': 1,
                      'model': {'name': 'mlp'}}
        assert(hu.hash_dict(exp_dict_1) == hu.hash_dict(exp_dict_2))
        assert(hu.hash_dict(exp_dict_1) != hu.hash_dict(exp_dict_3))

    def test_checkpoint(self):
        savedir_base = '.results'
        # create exp folder
        exp_dict = {'model': {'name': 'mlp', 'n_layers': 30},
                    'dataset': 'mnist', 'batch_size': 1}
        savedir = os.path.join(savedir_base, hu.hash_dict(exp_dict))
        hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
        hu.torch_save(os.path.join(savedir, "model.pth"), torch.zeros(10))
        hu.torch_load(os.path.join(savedir, "model.pth"))
        assert(os.path.exists(savedir))

        # delete exp folder
        hc.delete_experiment(savedir)
        assert(not os.path.exists(savedir))

        # check backup folder
        os.rmdir(savedir_base)

    def test_get_score_lists(self):
        # save a score_list
        savedir_base = '.tmp'
        exp_dict = {'model': {'name': 'mlp', 'n_layers': 30},
                    'dataset': 'mnist', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.5}, {'epoch': 0, 'acc': 0.9}]
        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                 'score_list.pkl'), score_list)
        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                  'exp_dict.json'), exp_dict)

        # check if score_list can be loaded and viewed in pandas
        exp_list = hr.get_exp_list(savedir_base=savedir_base)
        score_lists = hr.get_score_lists(exp_list, savedir_base=savedir_base)
        assert(score_lists[0][0]['acc'] == 0.5)
        assert(score_lists[0][1]['acc'] == 0.9)

        shutil.rmtree(savedir_base)

    def test_get_score_df(self):
        # save a score_list
        savedir_base = '.tmp'
        exp_dict = {'model': {'name': 'mlp', 'n_layers': 30},
                    'dataset': 'mnist', 'batch_size': 1}
        exp_dict2 = {'model': {'name': 'mlp2', 'n_layers': 30},
                     'dataset': 'mnist', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.5}, {'epoch': 0, 'acc': 0.9}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                 'score_list.pkl'), score_list)

        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                     'exp_dict.json'), exp_dict)

        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict2),
                     'exp_dict.json'), exp_dict)
        # check if score_list can be loaded and viewed in pandas
        exp_list = hr.get_exp_list(savedir_base=savedir_base)

        score_df = hr.get_score_df(exp_list, savedir_base=savedir_base)
        assert(np.array(score_df['dataset'])[0] == 'mnist')

        shutil.rmtree('.tmp')

    def test_get_plot(self):
        # save a score_list
        savedir_base = '.tmp'
        exp_dict = {'model': {'name': 'mlp', 'n_layers': 30},
                    'dataset': 'mnist', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.5}, {'epoch': 1, 'acc': 0.9}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                 'score_list.pkl'), score_list)
        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                  'exp_dict.json'), exp_dict)
        # check if score_list can be loaded and viewed in pandas
        exp_list = hr.get_exp_list(savedir_base=savedir_base)

        fig, axis = hr.get_plot(exp_list, savedir_base=savedir_base,
                                x_metric='epoch', y_metric='acc')

        fig.savefig(os.path.join('.tmp', 'test.png'))
        shutil.rmtree('.tmp')

    def test_get_result_manager(self):
        # save a score_list
        savedir_base = '.tmp_plots'
        if os.path.exists(savedir_base):
            shutil.rmtree(savedir_base)
        exp_dict = {'model': {'name': 'mlp', 'n_layers': 30},
                    'dataset': 'mnist', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.5}, {'epoch': 1, 'acc': 0.9}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                 'score_list.pkl'), score_list)
        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                  'exp_dict.json'), exp_dict)

        exp_dict = {'model': {'name': 'mlp', 'n_layers': 30},
                    'dataset': 'cifar10', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.25}, {'epoch': 1, 'acc': 1.24},
                      {'epoch': 2, 'acc': 1.5}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                 'score_list.pkl'), score_list)
        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                     'exp_dict.json'), exp_dict)

        exp_dict = {'model': {'name': 'lenet', 'n_layers': 30},
                    'dataset': 'cifar10', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.35}, {'epoch': 1, 'acc': 1.2},
                      {'epoch': 2, 'acc': 1.3}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                 'score_list.pkl'), score_list)
        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                  'exp_dict.json'), exp_dict)

        exp_dict = {'model': {'name': 'lenet', 'n_layers': 30},
                    'dataset': 'cifar10', 'batch_size': 5}

        score_list = [{'epoch': 0, 'acc': 0.15}, {'epoch': 1, 'acc': 1.21},
                      {'epoch': 2, 'acc': 1.7}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                 'score_list.pkl'), score_list)
        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                                  'exp_dict.json'), exp_dict)

        rm = hr.ResultManager(savedir_base=savedir_base,
                              groupby_list=['dataset'])

        assert(len(rm.exp_groups) == 2)
        for exp_list in rm.exp_groups:
            assert(exp_list[0]['dataset'] in ['mnist', 'cifar10'])
        rm.get_score_df()
        rm.get_score_lists()
        rm.get_images()

        fig_list = rm.get_plot(x_metric='epoch', y_metric='acc',
                               title_list=['dataset'], legend_list=['model'])
        for i, fig in enumerate(fig_list):
            fig.savefig(os.path.join(savedir_base, '%d.png' % i))

        order = 'groups_by_metrics'
        fig_list = rm.get_plot_all(
            order=order, x_metric='epoch', y_metric_list=['acc', 'epoch'],
            title_list=['dataset'], legend_list=['model'],
            log_metric_list=['acc'],
            map_exp_list=[({'batch_size': 5, 'model': {'name': 'lenet'}},
                           {'color': 'yellow', 'marker': '+',
                            'label': 'LeNet'})],
            map_title_list=[{'mnist': 'MNIST'}, {'cifar10': 'CIFAR-10'}],
            map_xlabel_list=[{'epoch': 'EPOCHS'}],
            map_ylabel_list=[{'acc': 'Score'}])

        for i, fig in enumerate(fig_list):
            fig.savefig(os.path.join(savedir_base, '%s_%d.png' % (order, i)))

        order = 'metrics_by_groups'
        fig_list = rm.get_plot_all(
            order=order, x_metric='epoch', y_metric_list=['acc', 'epoch'],
            title_list=['dataset'], legend_list=['model'], avg_across='model')
        for i, fig in enumerate(fig_list):
            fig.savefig(os.path.join(savedir_base, '%s_%d.png' % (order, i)))

        # shutil.rmtree(savedir_base)

    def test_filter_exp_list(self):
        exp_list = hu.cartesian_exp_group(
            {'dataset': ['imagenet', 'mnist', 'cifar10'], 'model': 'mlp',
             'batch_size': [1, 5]})

        exp_list = hr.filter_exp_list(
            exp_list, filterby_list=[{'dataset': 'mnist'},
                                     {'dataset': 'cifar10'}])
        visited = []
        for exp_dict in exp_list:
            assert(exp_dict['dataset'] in ['mnist', 'cifar10'])
            visited += [exp_dict['dataset']]

        assert('mnist' in visited)
        assert('cifar10' in visited)

    def test_group_exp_list(self):
        exp_list = hu.cartesian_exp_group(
            {'dataset': ['imagenet', 'mnist', 'cifar10'], 'model': 'mlp',
             'batch_size': [1, 5]})

        list_of_exp_list = hr.group_exp_list(exp_list, groupby_list='dataset')
        for exp_list in list_of_exp_list:
            assert (len(set([exp_dict['dataset'] for exp_dict in exp_list]))
                    == 1)

    def test_get_best_exp_dict(self):
        savedir_base = '.tmp'
        exp_dict_1 = {'model': {'name': 'mlp', 'n_layers': 30},
                      'dataset': 'mnist', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.5}, {'epoch': 1, 'acc': 0.9}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict_1),
                                 'score_list.pkl'), score_list)

        exp_dict_2 = {'model': {'name': 'mlp2', 'n_layers': 30},
                      'dataset': 'mnist', 'batch_size': 1}
        score_list = [{'epoch': 0, 'acc': 0.5}, {'epoch': 1, 'acc': 1.2}]
        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict_2),
                                 'score_list.pkl'), score_list)

        exp_list = [exp_dict_1, exp_dict_2]
        best_exp_dict = hr.get_best_exp_dict(
            exp_list, savedir_base=savedir_base, metric='acc',
            min_or_max='max')

        assert(best_exp_dict['model']['name'] == 'mlp2')

    def test_get_images(self):
        pass

    def test_get_job_stats_logs_errors(self):
        return
        exp_list = [{'model': {'name': 'mlp', 'n_layers': 30},
                    'dataset': 'mnist', 'batch_size': 1}]

        savedir_base = '/mnt/datasets/public/issam/tmp'
        job_config = {'volume': ['/mnt:/mnt'],
                      'image': 'images.borgy.elementai.net/issam.laradji/main',
                      'bid': '1',
                      'restartable': '1',
                      'gpu': '1',
                      'mem': '20',
                      'cpu': '2',
                      'username': 'issam'}
        run_command = ('python example.py -ei <exp_id> -sb %s'
                       % (savedir_base))
        try:
            hjb.run_exp_list_jobs(
                exp_list, savedir_base=savedir_base,
                workdir=os.path.dirname(os.path.realpath(__file__)),
                run_command=run_command,
                job_config=job_config,
                force_run=True,
                wait_seconds=0)
            assert (os.path.exists(os.path.join(savedir_base,
                                   hu.hash_dict(exp_list[0]),
                                   'borgy_dict.json')))
            stats = hr.get_job_stats(exp_list, savedir_base=savedir_base,
                                     username='issam')
            logs = hr.get_job_logs(exp_list, savedir_base=savedir_base,
                                   username='issam')
            errors = hr.get_job_errors(exp_list, savedir_base=savedir_base,
                                       username='issam')
        except:
            pass


if __name__ == '__main__':
    unittest.main()

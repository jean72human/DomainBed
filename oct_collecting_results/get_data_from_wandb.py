# %%

import wandb
import numpy as np

api = wandb.Api()


# %%
runs = api.runs("remix_school-of-rock/aengus-spawrious-comprehensive-runs")
runs2 = api.runs("remix_school-of-rock/aengus-spawrious-comprehensive-runs-round2")
len(runs)
len(runs2)
len(runs) + len(runs2)

# %%

# %%

# %%
s = 'alg-ERM_it-0_arch-resnet50-nopretraining_ds-SpawriousO2O_easy'
def get_run_info_from_name(s:str) -> dict:
    split = s.split('_')
    split[-2] = split[-2] + '_' + split[-1]
    split.pop()
    run_info = {}
    for key_value in split:
        if len(key_value.split('-')) > 2:
            key = key_value.split('-')[0]
            value = key_value.split('-')[1] + '-' + key_value.split('-')[2]
        else:
            key, value = key_value.split('-')
        run_info[key] = value
    return run_info

# %%
output_dir = 'ERM-resnet50-nopretraining-LISA-Mixup-results'
def get_da_info_from_output_dir(output_dir:str) -> (bool, str, str):
    da_bool = True
    da_strategy = output_dir.split('-')[-3]
    da_interp = output_dir.split('-')[-2]
    if da_strategy == 'no':
        da_bool = False

    return da_bool, da_strategy, da_interp


def get_early_stopping_test_acc(run):

    test_acc = 'failed'

    def helper():
        env0_in_acc_last = 0
        env1_in_acc_last = 0
        env2_in_acc_last = 0
        env0_out_acc_last = 0
        env1_out_acc_last = 0
        env2_out_acc_last = 0

        best_validation_acc = 0
        test_acc = 0
        for row in run.scan_history():
            if row['env0_in_acc'] is not None:
                env0_in_acc_last = row['env0_in_acc']
            else:
                row['env0_in_acc'] = env0_in_acc_last
            if row['env1_in_acc'] is not None:
                env1_in_acc_last = row['env1_in_acc']
            else:
                row['env1_in_acc'] = env1_in_acc_last
            if row['env2_in_acc'] is not None:
                env2_in_acc_last = row['env2_in_acc']
            else:
                row['env2_in_acc'] = env2_in_acc_last
            if row['env0_out_acc'] is not None:
                env0_out_acc_last = row['env0_out_acc']
            else:
                row['env0_out_acc'] = env0_out_acc_last
            if row['env1_out_acc'] is not None:
                env1_out_acc_last = row['env1_out_acc']
            else:
                row['env1_out_acc'] = env1_out_acc_last
            if row['env2_out_acc'] is not None:
                env2_out_acc_last = row['env2_out_acc']
            else:
                row['env2_out_acc'] = env2_out_acc_last
            
            validation_acc = np.mean([row['env1_in_acc'], row['env2_in_acc']])
            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                test_acc = row['env0_in_acc']
        return test_acc
        
    # set a 2 minute timer, then break if the timer if up
    import time
    start_time = time.time()
    while True:
        if time.time() - start_time > 120:
            break
        try:
            test_acc = helper()
        except:
            continue
        break
    
    return test_acc

# %%


count = 0
pre_integration_run_dict_list = []
for run in runs:
    if run.state != 'finished':
        continue
    # summary = run.summary
    # history = run.history()
    count += 1
    # if count > 10:
    #     break
    print(f"count: {count}")
    run_dict = get_run_info_from_name(run.name)
    da_bool, da_strategy, da_interp = get_da_info_from_output_dir(run.config['output_dir'])
    if da_bool:
        run_dict['alg'] = da_strategy + '-' + da_interp
    # run_dict['test_results'] = []
    # run_dict['test_results'].append(run.summary['env0_out_acc'])
    # run_dict['test_results'].append(run.summary['env0_in_acc'])
    print('about to look at this run_dict')
    print(run_dict)
    test_acc = get_early_stopping_test_acc(run)
    if test_acc == 'failed':
        continue
    run_dict['test_acc'] = test_acc
    pre_integration_run_dict_list.append(run_dict)

for run in runs2:
    if run.state != 'finished':
        continue
    # summary = run.summary
    # history = run.history()
    count += 1
    # if count > 10:
    #     break
    print(f"count: {count}")
    run_dict = get_run_info_from_name(run.name)
    da_bool, da_strategy, da_interp = get_da_info_from_output_dir(run.config['output_dir'])
    if da_bool:
        run_dict['alg'] = da_strategy + '-' + da_interp
    # run_dict['test_results'] = []
    # run_dict['test_results'].append(run.summary['env0_out_acc'])
    # run_dict['test_results'].append(run.summary['env0_in_acc'])
    print('about to look at this run_dict')
    print(run_dict)
    test_acc = get_early_stopping_test_acc(run)
    if test_acc == 'failed':
        continue
    run_dict['test_acc'] = test_acc
    pre_integration_run_dict_list.append(run_dict)

print(f"count: {count}")

print(f"count: {count}")
# %%


# Create the dictionaries that I will need
integrated_run_list = []
datasets =["SpawriousO2O_easy", "SpawriousO2O_medium", "SpawriousO2O_hard", "SpawriousM2M_easy", "SpawriousM2M_medium", "SpawriousM2M_hard"]
algs = ['ERM', 'GroupDRO', 'MMD', 'LISA-Mixup', 'LISA-CutMix', 'random_shuffle-CutMix', 'random_shuffle-Mixup']
da_algs = ['LISA-Mixup', 'LISA-CutMix', 'random_shuffle-CutMix', 'random_shuffle-Mixup']
archs = ['resnet50-nopretraining', 'vit-b']

for dataset in datasets:
    for alg in algs:
        for arch in archs:
            run_dict = {}
            run_dict['configs'] = {
                'ds': dataset,
                'alg': alg,
                'arch': arch
            }
            run_dict['test_acc'] = []
            run_dict['n_iterations'] = 0

            integrated_run_list.append(run_dict)

for alg in da_algs:
    for dataset in datasets:
        run_dict = {}
        run_dict['configs'] = {
            'ds': dataset,
            'alg': alg,
            'arch': 'resnet50'
        }
        run_dict['test_acc'] = []
        run_dict['n_iterations'] = 0

        integrated_run_list.append(run_dict)

for run_dict in pre_integration_run_dict_list:
    for integrated_run in integrated_run_list:
        if run_dict['alg'] == integrated_run['configs']['alg'] and run_dict['ds'] == integrated_run['configs']['ds'] and run_dict['arch'] == integrated_run['configs']['arch']:
            print(integrated_run['configs'])
            integrated_run['test_acc'].append(run_dict['test_acc'])
            integrated_run['n_iterations'] += 1

    
# %%
# make a plot where on the x axis we have n_iterations and on the y axis we have frequency in the list
import numpy as np
n_iterations_x = np.zeros(6)
for integrated_run in integrated_run_list:
    n_iterations_x[integrated_run['n_iterations']] += 1    

# now plot np.range(5) against n_iterations_x
import matplotlib.pyplot as plt
plt.bar(np.arange(6), n_iterations_x)
plt.xlabel('Number of Iterations')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Iterations')
plt.show()
# %%

# Write integrated_run_list to a jsonl file
import json
with open('integrated_run_list.jsonl', 'w') as f:
    for item in integrated_run_list:
        f.write(json.dumps(item) + '\n')
# %%
vitb_completed_count = 0
for run_dict in integrated_run_list:
    run_dict['iterations_left'] = 0
    if run_dict['n_iterations'] < 3:


        run_dict['iterations_left'] = 3 - run_dict['n_iterations']
        # print(run_dict)
        if run_dict['configs']['arch'] != 'vit-b' and run_dict['n_iterations'] == 0:
            print(run_dict)
    if run_dict['configs']['arch'] == 'vit-b':
        vitb_completed_count += run_dict['n_iterations']

print(vitb_completed_count)
    
# %%
import json
with open('combined_integrated_run_list.jsonl', 'w') as f:
    for item in integrated_run_list:
        f.write(json.dumps(item) + '\n')
# %%

jsonl_data = []
with open('combined_integrated_run_list.jsonl', 'r') as f:
    for line in f:
        jsonl_data.append(json.loads(line))
    
# %%

table_data = []
for line in jsonl_data:
    data_dict = {}
    data_dict['alg'] = line['configs']['alg']
    data_dict['arch'] = line['configs']['arch']
    data_dict[line['configs']['ds']] = {}

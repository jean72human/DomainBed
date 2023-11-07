# %%

import json
with open('integrated_run_list.jsonl', 'r') as f:
    integrated_run_list = [json.loads(line) for line in f.readlines()]


# %%

vitb_commands = []
rest_commands = []

da_algs = ['LISA-CutMix', 'LISA-Mixup', 'random_shuffle-CutMix', 'random_shuffle-Mixup']
nonda_algs = ['ERM', 'GroupDRO', 'MMD']

for run_dict in integrated_run_list:
    if run_dict['iterations_left'] == 0:
        continue

    if run_dict['configs']['alg'] in da_algs:
        
        command = f"python -m domainbed.scripts.loop_arch_resnet --dataset {run_dict['configs']['ds']} --algorithm ERM --architecture {run_dict['configs']['arch']} --da_bool True --da_strategy {run_dict['configs']['alg']} --n_iter {run_dict['iterations_left']}"

        if run_dict['configs']['arch'] == 'vit-b':
            vitb_commands.append(command)
        else:
            rest_commands.append(command)

    elif run_dict['configs']['alg'] in nonda_algs:

        command = f"python -m domainbed.scripts.loop_arch_resnet --dataset {run_dict['configs']['ds']} --algorithm {run_dict['configs']['alg']} --architecture {run_dict['configs']['arch']} --da_bool False --n_iter {run_dict['iterations_left']}"

        if run_dict['configs']['arch'] == 'vit-b':
            vitb_commands.append(command)
        else:
            rest_commands.append(command)



# %%

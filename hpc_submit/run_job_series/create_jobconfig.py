import itertools

output_file = 'config.txt'

learning_rate = [0.001, 0.0001]
weight_gecay = [0, 0.1, 0.01]
use_data_augmentation = [True, False]

combinations = list(itertools.product(learning_rate, weight_gecay, use_data_augmentation))
config_strings = []

for combination in combinations:
    if combination[2]:
        augmentation = '--use_data_augmentation'
    else:
        augmentation = ''

    config_string = f'--learning_rate={combination[0]} --weight_decay={combination[1]} {augmentation}'
    config_strings.append(config_string)
    print(config_string)

with open(output_file, 'w') as f:
    for config_string in config_strings:
        f.write(config_string + '\n')

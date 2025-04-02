import itertools

output_file = 'config.txt'

label_type = ['--label_type=category', '--label_type=sealed']
use_data_augmentation = [True, False]

combinations = list(itertools.product(label_type, use_data_augmentation))
config_strings = []

for combination in combinations:
    if combination[1]:
        augmentation = '--use_data_augmentation'
    else:
        augmentation = ''

    label = combination[0]

    config_string = f'{label} {augmentation}'
    config_strings.append(config_string)
    print(config_string)

with open(output_file, 'w') as f:
    for config_string in config_strings:
        f.write(config_string + '\n')

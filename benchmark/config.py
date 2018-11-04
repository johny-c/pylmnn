import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BENCHMARK_DIR, 'dataset_params.yml')

DATASETS = [
    'iris',
    'olivetti_faces',
    'letters',
    'usps',
    'isolet',
    'mnist_deskewed'
]

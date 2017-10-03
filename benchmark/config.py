import os

BENCHMARK_DIR = os.path.split(__file__)[0]

CONFIG_FILE = os.path.join(BENCHMARK_DIR, 'dataset_params.yml')

DATASETS = [
            'iris', 'olivetti_faces', 'letters', 'usps', 'isolet',
            # 'mnist_deskewed'
            ]
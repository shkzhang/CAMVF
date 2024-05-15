config = {
    'default': {
        'seed': range(0, 5),
        'dataset': 'default',
        'task_id': 'test-dist',
        'epochs': 1000,
        'batch_size': 32,
        'lambda_epochs': 250,
        'lr': 0.0005,
        'alpha': 0.14,
        'beta': 0.5,
        'exp_eta': 0.99,
        'encode_dim': [128, 256, 512],
        'proj_dim': [512, 256, 128],
        'dataset_type': 'tvt',
        'train_type': 'base'

    },
    'qmf': {
        'model': 'qmf',
        'train_type': 'qmf'
    },
    'camvf': {
        'model': 'camvf'
    },
    'ecamvf': {
        'model': 'ecamvf',
    },
    'gcca': {
        'model': 'gcca',
        'batch_size': 64,
        'latent_dimensions': 32,
        'train_type': 'cca'
    },
    'kcca': {
        'model': 'kcca',
        'batch_size': 64,
        'latent_dimensions': 32,
        'train_type': 'cca'

    },
    'dcca': {
        'model': 'dcca',
        'batch_size': 64,
        'latent_dimensions': 32,
        'train_type': 'cca'

    },
    'dccae': {
        'model': 'dccae',
        'batch_size': 64,
        'latent_dimensions': 32,
        'train_type': 'cca'
    },

    'tmc': {
        'model': 'tmc',

    },
    'etmc': {
        'model': 'etmc',
    },
}
data_set_config = {

    'default': {
        'dataset_type': 'tvt',
        'root': './data',
        'classes': 2,
        'dims': [146, 49, 152, 152, 152],
        'name_list': ['US', 'PARA', 'CEUS', 'CEUS_PCA_DOWN', 'CEUS_PCA_UP'],
        'epochs': 1000,
        'lambda_epochs': 2500,

    },
    'OrganMNIST3D': {
        'root': './data/OrganMNIST3D',
        'epochs': 1000,
        'batch_size': 128,
        'lambda_epochs': 2000,
        'classes': 11,
        'dims': [512, 512, 512, 512],
        'name_list': ['64_axial', '64_sagittal', '64_coronal', '64_diagonal'],
    },
    'NoduleMNIST3D': {
        'root': './data/NoduleMNIST3D',
        'epochs': 1000,
        'batch_size': 128,
        'lambda_epochs': 1,
        'classes': 2,
        'dims': [512, 512, 512, 512],

        'name_list': ['64_axial', '64_sagittal', '64_coronal', '64_diagonal'],
    },
    'AdrenalMNIST3D': {
        'root': './data/AdrenalMNIST3D',
        'epochs': 1000,
        'batch_size': 128,
        'lambda_epochs': 2000,
        'classes': 2,
        'dims': [512, 512, 512, 512],

        'name_list': ['64_axial', '64_sagittal', '64_coronal', '64_diagonal'],
    },
    'FractureMNIST3D': {
        'root': './data/FractureMNIST3D',
        'epochs': 1000,
        'batch_size': 128,
        'lambda_epochs': 500,
        'classes': 3,
        'dims': [512, 512, 512, 512],

        'name_list': ['64_axial', '64_sagittal', '64_coronal', '64_diagonal'],
    },
    'EXCEL': {
        'dataset_type': 'tvt',
        'root': './data/excel',
        'classes': 2,
        'encode_dim': [128, 64, 32],
        'dims': [131, 19, 11, 3],
        'name_list': ['texture', 'shape', 'edge', 'genes', ],
        'epochs': 2000,
        'lambda_epochs': 2500,
    },
    'BREAST':{
        'dataset_type': 'tvt',
        'root': './data',
        'classes': 2,
        'encode_dim': [128, 64, 32],
        'dims': [146, 152, 152, 152],
        'name_list': ['US', 'CEUS', 'wash_in', 'wash_out', ],
        'epochs': 50,
        'lambda_epochs': 2500,
    }

}
config['default'].update(data_set_config[config['default']['dataset']])

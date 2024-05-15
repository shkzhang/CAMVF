import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cca_zoo.deep import architectures, DCCAE, DCCA_EY
from cca_zoo.linear import GCCA
from cca_zoo.nonparametric import KCCA

from model.mlp import SingleMlp, SingleLinear
from model.loss import SupConLoss

import lightning.pytorch as pl


class CCAModel:
    def __init__(self, latent_dimensions, dims, model_type):
        self.latent_dimensions = latent_dimensions
        self.model_type = model_type
        if model_type == 'dcca':
            encoders = [architectures.Encoder(latent_dimensions=self.latent_dimensions, feature_size=dim) for dim in
                        dims]
            self.cca_model = DCCA_EY(latent_dimensions=self.latent_dimensions, encoders=encoders)
        elif model_type == 'dccae':
            encoders = [architectures.Encoder(latent_dimensions=self.latent_dimensions, feature_size=dim) for dim in
                        dims]
            decoders = [architectures.Decoder(latent_dimensions=self.latent_dimensions, feature_size=dim, dropout=0.1)
                        for dim in
                        dims]

            self.cca_model = DCCAE(latent_dimensions=self.latent_dimensions, encoders=encoders, decoders=decoders)
        elif model_type == 'gcca':
            self.cca_model = GCCA(latent_dimensions=self.latent_dimensions,random_state=0)
        elif model_type == 'kcca':
            self.cca_model = KCCA(latent_dimensions=self.latent_dimensions,random_state=0)

    def fit(self, epochs, train_loader, test_loader):
        if self.model_type == 'gcca' or self.model_type == 'kcca':
            return self.fit_model(self.cca_model, train_loader)
        else:
            return self.fit_cca_deep_model(self.cca_model, epochs, train_loader, test_loader)

    def transform(self, data):
        if self.model_type == 'gcca' or self.model_type == 'kcca':
            np_data = self.cca_model.transform(data)
            return [torch.from_numpy(x.astype(np.float32)).to(data[0].device) for x in np_data]
        else:
            views_device = [view.cuda() for view in data]
            return self.cca_model(views_device)

    def fit_model(self, cca_model, train_loader):
        data_dict = {}
        for _, data in enumerate(train_loader):
            for view in range(len(data['views'])):
                if view not in data_dict:
                    data_dict[view] = data['views'][view].cpu().numpy()
                else:
                    data_dict[view] = np.vstack((data_dict[view], data['views'][view].cpu().numpy()))
        cca_model.fit(data_dict.values())
        return self.cca_model

    @staticmethod
    def fit_cca_deep_model(cca_model, epochs, train_loader, test_loader):
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='val/objective',  # Choose the metric to monitor (validation loss in this case)
            mode='min',  # 'min' means stopping when the monitored quantity has stopped decreasing
            patience=4,  # Number of epochs with no improvement after which training will be stopped
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val/objective',
            mode='min',
            dirpath='ckpt',
            filename='best_model'
        )
        pl.seed_everything(0)
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[checkpoint_callback, early_stopping_callback],
            enable_checkpointing=True,
            enable_model_summary=False,
            enable_progress_bar=False,


        )
        trainer.fit(cca_model, train_dataloaders=train_loader, val_dataloaders=test_loader)

        # Load the best model
        cca_model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
        cca_model.eval()
        cca_model.freeze()
        cca_model.cuda()
        return cca_model


class CCABase(nn.Module):
    def __init__(self, cca_model, views, classes, latent_dimensions=32, encode_dim=[128, 256], proj_dim=[256, 128]):
        """
        :param encode_dim: the dim of feature information vector, list
        :param proj_dim: the hidden dim of projection module, list
        """
        super(CCABase, self).__init__()
        self.encode_dim = encode_dim
        self.proj_dim = proj_dim
        self.latent_dimensions = latent_dimensions
        self.classes = classes
        self.cca_model = cca_model

        self.FeatureInforEncoder = SingleMlp(latent_dimensions * views, self.encode_dim)
        self.Projection = SingleLinear(self.encode_dim[-1], self.proj_dim[0])
        self.ModalityClassifier = SingleLinear(self.encode_dim[-1], self.classes)
        self.softplus = nn.Softplus()

    def forward(self, x, y, global_step, mode='train'):
        assert mode in ['train', 'valid', 'test']
        evidence, project = self.infer(x)
        loss = 0

        inter_criterion = SupConLoss()
        inter_loss = inter_criterion(project, y)
        loss += inter_loss
        cross_criterion = nn.CrossEntropyLoss()
        loss += cross_criterion(evidence, y)

        return None, evidence, loss

    def infer(self, input):
        FeatureInfo = self.FeatureInforEncoder(input)
        ProjectVector = F.normalize(self.Projection(FeatureInfo), p=2, dim=1)
        Evidence = self.softplus(self.ModalityClassifier(FeatureInfo))
        return Evidence, ProjectVector

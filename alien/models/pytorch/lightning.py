"""Pytorch Lightning trainer."""

from typing import Optional
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ...data import TupleDataset
from ...decorators import get_defaults_from_self
from ...models import Classifier, Output, Regressor
from ...utils import convert_output_type, update_copy
from .training_limits import TrainingLimit, default_limit, get_training_limit


class DefaultLightningModule(pl.LightningModule):
    # pylint: disable=unused-argument
    def __init__(
        self,
        model,
        loss=F.mse_loss,
        optimizer="adam",
        al_model=None,
        wrapped_output=None,
        output=None,
        **opt_args,
    ):
        super().__init__()
        self.model = model
        self.al_model = al_model
        self.loss = loss
        self.optimizer = optimizer
        self.opt_args = opt_args
        self.wrapped_output = wrapped_output
        self.output = output

    def forward(self, X, *args, **kwargs):
        return self.model(X, *args, **kwargs)

    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), **self.opt_args)
        raise ValueError(f"Optimizer '{self.optimizer}' not supported.")

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        if self.al_model is not None:
            output = self.al_model._forward(X)  # pylint: disable=protected-access
        else:
            output = self.model(X)
        if self.output is not None and self.wrapped_output is not None:
            output = convert_output_type(X, self.wrapped_output, self.output)
        loss = self.loss(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        if self.al_model is not None:
            output = self.al_model._forward(X)  # pylint: disable=protected-access
        else:
            output = self.model(X)
        if self.output is not None and self.wrapped_output is not None:
            output = convert_output_type(X, self.wrapped_output, self.output)
        loss = self.loss(output, y)
        self.log("val_loss", loss)
        return loss


def get_dataloader(X=None, y=None, batch_size=1, collate_fn=None, **kwargs):
    if y is None:
        if isinstance(X, torch.utils.data.DataLoader):
            return X
        if isinstance(X, torch.utils.data.Dataset):
            return torch.utils.data.DataLoader(X, batch_size=batch_size, collate_fn=collate_fn, **kwargs)
        if isinstance(X, TupleDataset) and len(X.data) == 2:
            return torch.utils.data.DataLoader(X, batch_size=batch_size, collate_fn=collate_fn, **kwargs)
        # special case: designed to work with ListData
        if not isinstance(X, TupleDataset) and len(X.data) != 2: 
            return torch.utils.data.DataLoader(X, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

        # includes cases where X is a Dataset or a tensor/array of some sort
        return torch.utils.data.DataLoader(
            TupleDataset((X[..., :-1], X[..., -1])),
            collate_fn=collate_fn,
            **kwargs
        )
    return torch.utils.data.DataLoader(TupleDataset((X, y)), batch_size=batch_size, collate_fn=collate_fn, **kwargs)

class CustomWriter(pl.callbacks.BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # predictions = pl_module.all_gather(predictions)
        # batch_indices = pl_module.all_gather(batch_indices)

        # if pl_module.global_rank == 0: 
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{pl_module.global_rank}.pt"))
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{pl_module.global_rank}.pt"))


class LightningTrainer:
    """Pytorch Lightning trainer."""

    @get_training_limit
    def __init__(
        self,
        model,
        training_limit: Optional[TrainingLimit] = default_limit,
        batch_size=None,
        collate_fn=None,
        deterministic=True,
        al_model=None,
        loss=None,
        wrapped_output=None,
        output=None,
        **kwargs,
    ):
        if isinstance(model, pl.LightningModule):
            self.model = model
        elif isinstance(model, torch.nn.Module):
            if loss is None:
                if isinstance(al_model, Classifier):
                    loss = F.cross_entropy
                elif isinstance(al_model, Regressor):
                    loss = F.mse_loss
            if loss is F.cross_entropy:
                output = Output.LOGIT
            if al_model is not None and hasattr(al_model, "wrapped_output"):
                wrapped_output = al_model.wrapped_output
            else:
                wrapped_output = None
            self.model = DefaultLightningModule(
                model, al_model=al_model, loss=loss, wrapped_output=wrapped_output, output=output
            )

        self.training_limit = training_limit
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        kwargs["deterministic"] = deterministic
        self.al_model = al_model
        self.kwargs = kwargs
        self.loss = loss

    @get_training_limit
    @get_defaults_from_self
    def fit(
        self,
        X,
        y,
        batch_size=64,
        val_loader=None,
        collate_fn=None,
        training_limit=None,
        **kwargs,
    ):
        train_loader = get_dataloader(X, y, batch_size=batch_size, collate_fn=collate_fn)
        training_limit = training_limit if training_limit is not None else self.training_limit
        steps = training_limit.batch_limit(batch_size=train_loader.batch_size, length=len(train_loader.dataset))

        new_kwargs = update_copy(self.kwargs, min_steps=steps, max_steps=steps, **kwargs)
        pl.Trainer(**new_kwargs).fit(self.model, train_loader, val_loader)

    def predict(
        self, 
        X, 
        collate_fn=None,
        out_dir=None,
        **kwargs, 
    ): 
        """Return the prediction of a given batch"""
        pred_loader = get_dataloader(X, None, collate_fn=collate_fn, num_workers=4) 
        if out_dir: 
            pred_writer = CustomWriter(output_dir=out_dir, write_interval="epoch")
            self.pred_trainer = pl.Trainer(enable_progress_bar=False, callbacks=[pred_writer], **kwargs)
            self.pred_trainer.predict(self.model, pred_loader, return_predictions=False)
            # self.pred_trainer.strategy.barrier() # wait for all the process to finish writing to file
            # trainer.strategy.teardown()
        else: 
            raise NotImplementedError("Issue with prediction with different size (can't sync padding size across instances)")
            trainer = pl.Trainer(enable_progress_bar=False, **kwargs)
            predictions = trainer.predict(self.model, pred_loader, return_predictions=True)
            print(predictions.device)
            predictions_gathered = self.model.all_gather(predictions)
            if self.model.global_rank == 0: 
                return torch.cat(predictions_gathered)
    
    # for ddp 
    def barrier(self): 
        """make sure all rank finish processing before continuing"""
        self.pred_trainer.strategy.barrier()

    def get_rank(self): 
        return self.model.global_rank
    
    
        
            
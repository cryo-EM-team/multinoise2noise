import lightning as pl


class EpochInformerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Inform the datamodule about the current epoch
        if hasattr(trainer.datamodule, "current_epoch"):
            trainer.datamodule.current_epoch = trainer.current_epoch
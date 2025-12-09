import pytorch_lightning as pl


class LogBestModelPath(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        best_model_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.best_model_path:
                best_model_path = cb.best_model_path
                break
        if best_model_path is not None:
            if trainer.logger:
                trainer.logger.log_metrics({"best_model_path": best_model_path})
            # print(f"Best model path: {best_model_path}")

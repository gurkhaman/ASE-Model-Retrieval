import wandb
import numpy as np


class WandBLogger:
    def __init__(self, project, run_name=None, config=None, group=None, mode="online"):
        self.enabled = True
        try:
            wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                group=group,
                mode=mode,  # use "disabled" to silence logging
            )
        except Exception as e:
            print(f"[WandBLogger] Failed to init WandB: {e}")
            self.enabled = False

    def log_metrics(self, summary_dict, step=None):
        if not self.enabled:
            return
        wandb.log(summary_dict, step=step)

    def log_histogram(self, name, values, step=None):
        if not self.enabled:
            return
        wandb.log({name: wandb.Histogram(np.array(values))}, step=step)

    def log_config(self, config_dict):
        if not self.enabled:
            return
        wandb.config.update(config_dict)

    def log_table(self, name, columns, data):
        if not self.enabled:
            return
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})

    def finish(self):
        if self.enabled:
            wandb.finish()
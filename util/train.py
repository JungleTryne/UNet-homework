from pytorch_lightning import Trainer
from util.lightning import LightningUNet

from util.constants import DEFAULT_CHECKPOINT_PATH


def train(device, checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    lightning_model = LightningUNet()

    trainer = Trainer(
        gpus=1,
        accelerator=device,
        max_epochs=7,
        gradient_clip_val=1.0,
        val_check_interval=0.5,
        enable_checkpointing=True,
        default_root_dir='./bin'
    )

    trainer.fit(lightning_model)
    trainer.save_checkpoint(checkpoint_path)

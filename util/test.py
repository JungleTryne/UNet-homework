from util.lightning import LightningUNet
from pytorch_lightning import Trainer


def test(checkpoint_path, device):
    lightning_model = LightningUNet.load_from_checkpoint(
        checkpoint_path=checkpoint_path)

    trainer = Trainer(
        gpus=1,
        max_epochs=1,
        accelerator=device,
        default_root_dir='./bin'
    )

    trainer.test(lightning_model)

# U-Net for image segmentation

Blazingly fast (🚀🚀🚀) and super inaccurate (😈😈😈) U-Net with ResNet bottleneck components for image segmentation 

## How to install?

Create new virtual enviroment and execute
```
pip3 install -r requirements.txt
```

## How to use?

For testing the model, simply do

```
python3 main.py --test --device cuda
```

When using the model the first time it might ask you to download the dataset and the checkpoint. Without those components the model is unusable.

## How to train?

Training can be done with the following command:
```
python3 main.py --train --device cuda --checkpoint_path ./bin/checkpoint/my_checkpoint.ckpt
```

## Perfomance
I wanna cry
```
╔════════════╦═════════╗
║   Metric   ║  Value  ║
╠════════════╬═════════╣
║ Dice Score ║  0.0042 ║
║ MSE        ║ 5255.99 ║
╚════════════╩═════════╝
```

### Author
Danila Mishin

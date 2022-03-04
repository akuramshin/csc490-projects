# Module 1: Object Detection

This repository contains the starter code for Assignment 1 of CSC490H1.
In this assignment, you will implement a basic object detector for self-driving.

## How to run

### Overfitting

To overfit the detector to a single frame of PandaSet, run the following command
from the root directory of this repository:

```bash
python -m detection.main overfit --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs> --loss_func=<heatmap_loss_function> --kernel=<kernel>
```

This command will write model checkpoints and visualizations to `<your_path_to_outputs>`.

`<heatmap_loss_function>` is the loss function in the process of building heatmap, in the format of `<name>_<hyperparameters>`. `<name>` should be `mse`(default), `focal`
or `abfocal` which means MSE loss, Focal loss and alpha balanced focal loss respectively. Focal loss only has one hyperparameter $\gamma$, so it is in the format `focal_<gamma>`, e.g. focal_1. Alpha balanced loss has 2 hyperparameters, $\alpha$ and $\gamma$. It is in the format of `abfocal_<alpha>_<gamma>`, e.g. abfocal_0.5_0.

`<kernel>` means the type of gaussian kernel applied. It should be `iso`(default), `aniso`,  `rotate`, which indicates isotropic, anisotropic, rotated kernel respectively. 

### Training

To train the detector on the training split, run the following command
from the root directory of this repository:

```bash
python -m detection.main train --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs> --loss_func=<heatmap_loss_function> --kernel=<kernel>
```

This command will write model checkpoints and visualizations to `<your_path_to_outputs>`.

`<heatmap_loss_function>` is the loss function in the process of building heatmap, in the format of `<name>_<hyperparameters>`. `<name>` should be `mse`(default), `focal`
or `abfocal` which means MSE loss, Focal loss and alpha balanced focal loss respectively. Focal loss only has one hyperparameter $\gamma$, so it is in the format `focal_<gamma>`, e.g. focal_1. Alpha balanced loss has 2 hyperparameters, $\alpha$ and $\gamma$. It is in the format of `abfocal_<alpha>_<gamma>`, e.g. abfocal_0.5_0.

`<kernel>` means the type of gaussian kernel applied. It should be `iso`(default), `aniso`,  `rotate`, which indicates isotropic, anisotropic, rotated kernel respectively. 

### Visualization

To visualize the detections of the detector, run the following command
from the root directory of this repository:

```bash
python -m detection.main test --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs> --checkpoint_path<your_path_to_checkpoint>
```

This command will save detection visualizations to `<your_path_to_outputs>`.

### Evaluation

To evaluate the detections of the detector, run the following command
from the root directory of this repository:

```bash
python -m detection.main evaluate --data_root=<your_path_to_dataset> --output_root=<your_path_to_outputs> --checkpoint_path<your_path_to_checkpoint>
```

This command will save detection visualizations and metrics to `<your_path_to_outputs>`.

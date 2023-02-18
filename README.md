# D-NeRF

Adapted from [D-NeRF](https://github.com/albertpumarola/D-NeRF), re-structured for better interpretability and extensibility.


## Train

```shell script
python train_nerf.py config/blender/mutant.yaml --use_wandb
python train_nerf.py config/blender/trex.yaml --use_wandb
```

## Test

```shell script
python test_nerf.py config/blender/mutant.yaml --checkpoint 799000
python test_nerf.py config/blender/trex.yaml --checkpoint 799000
```

| blender            | mutant | trex   | bouncingballs |
|--------------------|--------|--------|---------------|
| reported by paper  | 31.29  | 31.75  | 32.80         |
| official pre-trained weights | 30.91 | 31.04 | 38.18 |
| our implementation | 30.69  | 30.18 | 38.33         |
```
config.yaml  Main config
callbacks/   Misc options for the Trainer (see src/callbacks/)
dataset/     Instantiates a datamodule (see src/dataloaders/)
experiment/  Defines a full experiment (combination of all of the above configs)
loader/      Defines a PyTorch DataLoader
model/       Instantiates a model backbone (see src/models/)
optimizer/   Instantiates an optimizer
pipeline/    Combination of dataset/loader/task for convenience
scheduler/   Instantiates a learning rate scheduler
task/        Defines loss, metrics, optional encoder/decoder (see src/tasks/)
trainer/     Flags for the PyTorch Lightning Trainer class
generate/    Additional flags used by the generate.py script
```


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



Run experiments:
python train_safari.py train.debug=true +trainer.fast_dev_run=1


Train mnist using long-conv model:
python train_safari.py trainer.max_epochs=2 train.test=1 ++trainer.limit_train_batches=1 ++trainer.limit_val_batches=1 ++trainer.limit_test_batches=1 

Train LRA cifar10:
python train_safari.py experiment=lra/long-conv-lra-cifar trainer.max_epochs=2 train.test=1 ++trainer.limit_train_batches=1 ++trainer.limit_val_batches=1 ++trainer.limit_test_batches=1 


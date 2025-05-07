import time
import pytorch_lightning as pl
from types import SimpleNamespace
import mlflow
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from LitDeepSDMData import LitDeepSDMData
from LitUNetSDM import LitUNetSDM
import yaml

# load configurations
yaml_conf = './DeepSDM_conf.yaml'
with open(yaml_conf, 'r') as f:
    DeepSDM_conf = yaml.load(f, Loader = yaml.FullLoader)
DeepSDM_conf = SimpleNamespace(**DeepSDM_conf)

# Define timelog
timelog = time.strftime('%Y%m%d%H%M%S', time.localtime())

# ## LOGGER
# Use mlflow to auto logging pytorch lightning everything
# mlflow.set_experiment(conf.experiment_name)
# mlflow.pytorch.autolog(disable = True)

### DATA
# initialize the lightning data module
deep_sdm_data = LitDeepSDMData(yaml_conf = yaml_conf)

### MODEL
# initialize the DeepSDM lightning module
model = LitUNetSDM(yaml_conf = yaml_conf)

# Check the metrics monitored and save models (checkpoints)
model_checkpoint_conf = SimpleNamespace(**DeepSDM_conf.model_checkpoint_conf)
checkpoint_callback = ModelCheckpoint(
    save_last = model_checkpoint_conf.save_last,
    save_top_k = model_checkpoint_conf.save_top_k,
    verbose = model_checkpoint_conf.verbose,
    monitor = model_checkpoint_conf.monitor,
    mode = model_checkpoint_conf.mode
)

# setup early stopper for reaching a plateau
earlystopping_conf = SimpleNamespace(**DeepSDM_conf.earlystopping_conf)
early_stop_callback = EarlyStopping(
    monitor = earlystopping_conf.monitor,                                 
    min_delta = earlystopping_conf.min_delta, 
    patience = earlystopping_conf.patience,                                 
    verbose = earlystopping_conf.verbose, 
    mode = earlystopping_conf.mode)

### TRAINER (all kinds of loops)
# change the devices number if you have only 1 GPU or more GPUs
# We use half precision for less memory usage and faster calculations
trainer_conf = SimpleNamespace(**DeepSDM_conf.trainer_conf)
trainer = pl.Trainer(
    max_epochs = DeepSDM_conf.training_conf['epochs'], 
    devices = trainer_conf.devices, 
    accelerator = trainer_conf.accelerator, 
    check_val_every_n_epoch = trainer_conf.check_val_every_n_epoch,
    strategy = DDPStrategy(static_graph=True), # use 'ddp_fork_find_unused_parameters_true' instead on jupyter or colab
    precision = trainer_conf.precision, 
    callbacks = [checkpoint_callback, early_stop_callback], 
    logger = pl.loggers.MLFlowLogger(experiment_name = DeepSDM_conf.training_conf['experiment_name'], run_name = timelog, log_model = True)
)

# Start the training!
# with mlflow.start_run(run_name=f"{timelog}-{trainer.global_rank}"):
#     ### START
#     trainer.fit(model, datamodule=deep_sdm_data)
trainer.fit(model, datamodule = deep_sdm_data)


# run `mlflow ui` in the console at the path that contains 'mlruns' for monitoring the trainig

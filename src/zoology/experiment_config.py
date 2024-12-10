import uuid
import numpy as np
import pathlib
from zoology.config import TrainConfig, ModelConfig, DataConfig, DataSegmentConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "figure2" + sweep_id

WANDB_USERNAME = None  # KAW: wandb username is optional; defaults to the one you're logged in as 

VOCAB_SIZE = 8_192


configs = []
for input_seq_len, num_kv_pairs in [
    #(64, 4),
    (128, 8),
    #(256, 16),
    # (512, 64),
]:
    if input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 128
    elif input_seq_len == 256:
        batch_size = 256
    else:
        batch_size = 512


    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "train_power_a": 0.01,
        "test_power_a": 0.01,
        "random_non_queries": False
    }

    data = DataConfig(
        train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        batch_size=batch_size,
        cache_dir=str(pathlib.Path(__file__).parent.absolute() / "dataset_cache"),
    )

    for d_model in [
        64, 
        #128, 
        #256, 
        #512,
    ]:
        for d_conv in [4]:
            for expand in [2]:
                #for n_layers in [2, 4]:
                for n_layers in [2]:
                    for lr in np.logspace(-4, -2, 4)[-2:-1]:
                        MIXERS = {
                            "attention": dict(
                                name="zoology.mixers.attention.MHA",
                                kwargs=dict(
                                    dropout=0.1,
                                    num_heads=1,
                                ),
                            ),
                            "prnn": dict(
                                name="zoology.mixers.prnn.pRNN",
                                kwargs=dict(
                                    num_layers=n_layers,
                                    bias=True,
                                    batch_first=True,
                                    dropout=0.,
                                    bidirectional=False,
                                    num_iters=2,  # number of iterations for quasi-DEER
                                    method="minrnn",  # minrrn or gru
                                    parallel=True,  # parallel implementation
                                ),
                            )
                        }

                        for sequence_mixer in [
                            #"attention",
                            "prnn",
                        ]:
                            block_type = "TransformerBlock"

                            model = ModelConfig(
                                d_model=d_model,
                                n_layers=n_layers,
                                block_type=block_type,
                                max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                                vocab_size=VOCAB_SIZE,
                                sequence_mixer=MIXERS[sequence_mixer],
                                state_mixer=dict(name="torch.nn.Identity", kwargs={})
                            )
                            config = TrainConfig(
                                model=model,
                                data=data,
                                learning_rate=lr,
                                max_epochs=64,  # NOTE: make longer to ensure convergence
                                run_id=f"{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr}-kv{num_kv_pairs}",
                                logger=LoggerConfig(
                                    project_name="zoology",
                                    entity=WANDB_USERNAME,
                                )

                            )
                            configs.append(config)
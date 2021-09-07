from core.data import codenet_paths
from core.distributed import gcp

# Calculate number of TPUs needed for sweep.
n = 2

# Ensure TPUs are up and unused.
# gcp.tpu_up_n(n)
gcp.fix_firewall().wait()

access_token = codenet_paths.get_personal_access_token()

gcp.tpu_run_script(
    'scripts/setup-tpu.sh', n, {
        'PERSONAL_ACCESS_TOKEN': access_token
    }
)

gcp.tpu_run_command('time echo "Hello world"', n)


hparams = {
    'config.learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
    # 'config.rnn_layers': [2, 4]
    'config.hidden_size': [16, 32, 64, 128, 256, 512, 1024],
    'config.span_encoding_method': ['first', 'mean', 'max', 'sum'],
}


gcp.tpu_run_command(
    'cd compressive-ipagnn && '
    'python3 -m scripts.runner '
    '--config.model_class=Transformer '
    '--config.batch_size=8 '
    '--dataset_path=/mnt/runtime-error-problems-experiments/datasets/project-codenet/full-noudf-ids '
    '--config.eval_freq=50000 '
    '--config.eval_subsample=1 '
    '--config.eval_max_batches=2500 '
    '--config.save_freq=25000 ', 
    n
)

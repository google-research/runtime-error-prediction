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

gcp.tpu_run_command(
    'cd compressive-ipagnn && '
    'python3 -m scripts.runner '
    '--config.model_class=Transformer '
    '--config.batch_size=8 '
    '--dataset_path=/mnt/runtime-error-problems-experiments/datasets/project-codenet/full-noudf-ids '
    '--config.epochs=1 '
    '--config.eval_freq=50000 '
    '--config.eval_subsample=1 '
    '--config.eval_max_batches=2500 '
    '--config.save_freq=25000 ', 
    n
)

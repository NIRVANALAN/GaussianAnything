partition=ips_share
job_name=compile
gpus=1
g=$((${gpus}<8?${gpus}:8))


srun -u --partition=${partition} --job-name=${job_name} \
    -n1 --gres=gpu:${gpus} --ntasks-per-node=1 -w 'SH-IDC1-10-198-6-85' \
    python3 emd_module.py

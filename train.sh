# python dinov2/train/train.py \
# --config-file dinov2/configs/train/mammo-config.yaml \
# --output-dir /home/nikola.jovisic.ivi/nj/outputs-dino/ \
torchrun --nproc_per_node=1 dinov2/train/train.py \
    --config-file dinov2/configs/train/mammo-config.yaml \
    --output-dir /home/nikola.jovisic.ivi/nj/outputs-dino-06/ \

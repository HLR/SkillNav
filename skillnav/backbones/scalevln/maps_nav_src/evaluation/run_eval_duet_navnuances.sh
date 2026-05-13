# ANNTROOT="../baselines/VLN-DUET/datasets/R2R/annotations"
ANNTROOT="/home/matiany3/ScaleVLN/VLN-DUET/datasets/NavNuances/annoatations"
SUBMITROOT="/home/matiany3/ScaleVLN/VLN-DUET/datasets/NavNuances/eval/exprs_map/finetune/dagger-vit-b16-seed.0-init.aug.45k-direction-pretrained/preds"
# SUBMITROOT="/home/matiany3/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-aug.mp3d.prevalent.hm3d_gibson.envdrop.init.140k-aug.hm3d.envdrop/preds"
OUTROOT="/home/matiany3/ScaleVLN/VLN-DUET/datasets/NavNuances/eval/exprs_map/finetune/dagger-vit-b16-seed.0-init.aug.45k-direction-pretrained"
SCANDIR="/home/matiany3/MapGPT/datasets/R2R/connectivity/scans.txt"

python /home/matiany3/navnuances/evaluation/eval.py \
--annotation_root $ANNTROOT \
--submission_root $SUBMITROOT \
--out_root $OUTROOT \
--scans_dir $SCANDIR

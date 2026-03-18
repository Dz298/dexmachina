OBJ=ketchup
HAND=orca_hand
CLIP=${OBJ}-0-600
python retargeting/parallel_retarget.py --clip $CLIP --hand ${HAND} --control_steps 2000 --save_name para --save -ow 

HAND=orca_hand
FNAME=assets/arctic/processed/s01/ketchup_use_01.npy
python retargeting/map_contacts.py --hand $HAND --load_fname $FNAME 

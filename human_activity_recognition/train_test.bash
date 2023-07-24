#!/bin/bash

CHECKPOINT_FILE="./ckps/"

CMD=""
CMD="main.py \
    --checkpoint-file ${CHECKPOINT_FILE} \
    --mode train \
    --model lstm \
    --evaluation evaluate\"
    
    eval "python $CMD "

#!/bin/sh

#SBATCH --nodelist=barre
#SBATCH --mem=16G
#SBATCH --time=1800
#SBATCH --output=/home/s2063518/unite_talromur.log

NUM_TRAIN=0
NUM_EVAL=0
DATASET_NAME=Talromur-full

SCRATCH_HOME=/disk/ostrom/scratchdir/s2063518
PREP_HOME=$SCRATCH_HOME/fs_data/preprocessed

OUT_DIR=$PREP_HOME/$DATASET_NAME
touch $OUT_DIR/train.txt
touch $OUT_DIR/val.txt

grep -o . <<< "abcdefgh" | while read VOICE;  do
    if [  "$NUM_TRAIN" -eq "0" ]; then
        cat $PREP_HOME/Talromur-$VOICE/train.txt >> $OUT_DIR/train.txt
    else
        head $PREP_HOME/Talromur-$VOICE/train.txt -n $NUM_TRAIN >> $OUT_DIR/train.txt
    fi

    if [  "$NUM_EVAL" -eq "0" ]; then
        cat $PREP_HOME/Talromur-$VOICE/val.txt >> $OUT_DIR/val.txt
    else
        head $PREP_HOME/Talromur-$VOICE/val.txt -n $NUM_EVAL >> $OUT_DIR/val.txt
    fi
done
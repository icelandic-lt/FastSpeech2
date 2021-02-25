#!/bin/sh

#SBATCH --nodelist=barre
#SBATCH --mem=16G
#SBATCH --time=1800
##SBATCH --gres=gpu:1
#SBATCH --output=/home/s2063518/prep_talromur.log

source /home/${USER}/.bashrc
source activate fs2

SCRATCH_HOME=/disk/ostrom/scratchdir/s2063518
FS_HOME=/home/${USER}/projects/FastSpeech2

grep -o . <<< "bcd" | while read VOICE;  do
    echo "Preparing voice $VOICE"
    if [ ! -d $SCRATCH_HOME/talromur/$VOICE ]; then
        # Have not unarchived
        unzip $SCRATCH_HOME/talromur_zips/$VOICE.zip -d $SCRATCH_HOME/talromur/
    fi
    PREP_DIR=$SCRATCH_HOME/fs_data/preprocessed/Talromur-$VOICE
    TEXTGRID_DIR=$PREP_DIR/TextGrid
    if [ ! -d $PREP_DIR ]; then
        # Have to copy textgrids from MFA alignments to prepro directory
        mkdir $PREP_DIR
        mkdir $TEXTGRID_DIR
        cp $SCRATCH_HOME/alignments/$VOICE/audio/*.TextGrid $PREP_DIR/TextGrid
    fi

    if [ ! -f $PREP_DIR/train.txt ]; then
        # running preprocessing
        echo "Running FastSpeech2 preprocess"
        python -u $FS_HOME/preprocess.py $SCRATCH_HOME/talromur/$VOICE $PREP_DIR Talromur-$VOICE > /home/${USER}/prep_talromur_python.log
    fi
done

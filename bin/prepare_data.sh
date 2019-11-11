#!/bin/bash

SNLI_ID=1wkAjMu-Pqnm1l-92M7UEp5YEtT1cFgVz
QQP_TRAIN_ID=1dnck-CCIyx8y2xg1vwFzcwXieZJB7ERC
QQP_TEST_ID=1XD-HxzUCTHrzhfvIXOlgqN_MWiiAqM8h

SNLI_DATA=${SNLI_DATA:-https://drive.google.com/uc?export=download&id=${SNLI_ID}}
QQP_DATA_TRAIN=${QQP_DATA_TRAIN:-https://drive.google.com/uc?export=download&id=${QQP_TRAIN_ID}}
QQP_DATA_TEST=${QQP_DATA_TEST:-https://drive.google.com/uc?export=download&id=${QQP_TEST_ID}}
ANLI_DATA_LINK=https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip

CORPORA_DIR=corpora
SNLI_DIR=SNLI
QQP_DIR=QQP
ANLI_DIR=ANLI

SNLI_FILE=train_snli.tgz
QQP_FILE_TRAIN=qqp_train.tgz
QQP_FILE_TEST=qqp_test.tgz
ANLI_FILE=anli_v0.1.zip

mkdir ../${CORPORA_DIR}
cd ../${CORPORA_DIR}
mkdir ${SNLI_DIR} ${QQP_DIR} ${ANLI_DIR}

function google_drive_big_file_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

wget --no-check-certificate ${SNLI_DATA} -O ${SNLI_DIR}/${SNLI_FILE}
wget --no-check-certificate ${QQP_DATA_TRAIN} -O ${QQP_DIR}/${QQP_FILE_TRAIN}
google_drive_big_file_download ${QQP_TEST_ID} ${QQP_DIR}/${QQP_FILE_TEST}
wget --no-check-certificate ${ANLI_DATA_LINK} -O ${ANLI_DIR}/${ANLI_FILE}

tar -xvzf ${SNLI_DIR}/${SNLI_FILE} -C ${SNLI_DIR}
tar -xvzf ${QQP_DIR}/${QQP_FILE_TRAIN} -C ${QQP_DIR}
tar -xvzf ${QQP_DIR}/${QQP_FILE_TEST} -C ${QQP_DIR}
unzip ${ANLI_DIR}/${ANLI_FILE} -d ${ANLI_DIR}
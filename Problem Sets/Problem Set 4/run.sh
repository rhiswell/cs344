#!/bin/bash

set -e

PROJECT_HOME=~/Workspace/cs344
TARGET_HOST=730
BIN_NAME=HW4
CMD_ARGS="../Problem\\ Sets/Problem\\ Set\\ 4/red_eye_effect_5.jpg ../Problem\\ Sets/Problem\\ Set\\ 4/red_eye_effect_template_5.jpg"

ssh $TARGET_HOST "cd $PROJECT_HOME/build && make $BIN_NAME && \
    cd ../bin && ./$BIN_NAME $CMD_ARGS"


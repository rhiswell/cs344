#!/bin/bash

set -e

PROJECT_HOME=~/Workspace/cs344
TARGET_HOST=730
BIN_NAME=HW3
CMD_ARGS="../Problem\\ Sets/Problem\\ Set\\ 3/memorial_raw.png"

ssh $TARGET_HOST "cd $PROJECT_HOME/build && make $BIN_NAME && \
    cd ../bin && ./$BIN_NAME $CMD_ARGS"


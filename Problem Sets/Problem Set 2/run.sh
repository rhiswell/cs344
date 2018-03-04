#!/bin/bash

set -e

PROJECT_HOME=~/Workspace/cs344
TARGET_HOST=730
BIN_NAME=HW2
CMD_ARGS="../Problem\\ Sets/Problem\\ Set\\ 2/cinque_terre_small.jpg"

ssh $TARGET_HOST "cd $PROJECT_HOME/build && make $BIN_NAME && \
    cd ../bin && ./$BIN_NAME $CMD_ARGS"


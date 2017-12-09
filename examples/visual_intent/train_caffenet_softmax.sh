#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=models/visual_intent/caffenet/solver_softmax.prototxt --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel --gpu 2,3 $@

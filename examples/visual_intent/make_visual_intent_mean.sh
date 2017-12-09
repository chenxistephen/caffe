#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=lmdb/visual_intent_30
DATA=data/VisualIntent
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/visual_intent_train_lmdb \
  $DATA/visual_intent_mean.binaryproto

echo "Done."

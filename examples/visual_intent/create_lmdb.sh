#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

REDO=true
LMDBDIR=lmdb/visual_intent_30
DATA=data/VisualIntent/Annotations_multiclass
TOOLS=build/tools
CATEGORY_LIST=data/VisualIntent/taxonomy/visual_intent_30_labels.txt
DBPREFIX=visual_intent

TRAIN_DATA_ROOT=./data/VisualIntent/Images/
VAL_DATA_ROOT=./data/VisualIntent/Images/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset_multilabel \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $LMDBDIR/visual_intent_train_lmdb \
    $CATEGORY_LIST

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset_multilabel \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $LMDBDIR/visual_intent_val_lmdb \
    $CATEGORY_LIST

echo "Done."

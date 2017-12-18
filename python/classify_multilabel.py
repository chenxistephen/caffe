#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import matplotlib.pyplot as plt
import caffe
import os.path as osp

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)
    imgPath = '../VisualIntent/Images/'
    labelFile = '../VisualIntent/taxonomy/visual_intent_30_labels.txt'
    gtFile = '../VisualIntent/Annotations_multiclass/val.txt'
    imgsetFile = '../VisualIntent/ImageSets/val.txt'

    classes = [l.rstrip() for l in open(labelFile,'r').readlines()]
    
    gtlines = [l.rstrip() for l in open(gtFile,'r').readlines()]
    imglist = [l.split('\t')[0] for l in gtlines]
    imglabels = [l.split('\t')[1] for l in gtlines]
    name_label_map = dict(zip(imglist, imglabels))

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--input_file",
        default=imglist,
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "--output_dir",
        default='../visualizations',
        help="Output npy filename."
    )
    parser.add_argument(
        "--output_pred_file",
        default=None,
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../../trained_models/VisualIntent/caffenet/softmax_50k/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../../trained_models/VisualIntent/caffenet/softmax_50k/caffenet_softmax_iter_50000.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        default=True,
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default='', #os.path.join(pycaffe_dir,'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()
    PIXEL_MEANS = np.array([[[128.223/255.0,135.409/255.0,142.853/255.0]]])
    print PIXEL_MEANS
    
    output_dir = args.output_dir
    inputName = osp.splitext(osp.basename(imgsetFile))[0]
    # print inputName
    modelName = osp.splitext(osp.basename(args.pretrained_model))[0]
    print modelName
    visFolder = osp.join(output_dir, modelName)
    outresFolder = osp.join(output_dir, 'pred_results')
    output_pred_file = osp.join(outresFolder, modelName)
    if not osp.isdir(visFolder):
        os.makedirs(visFolder)
    if not osp.isdir(outresFolder):
        os.makedirs(outresFolder)

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]
    print mean
    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    # args.input_file = os.path.expanduser(args.input_file)
    # if os.path.isdir(args.input_file):
        # print("Loading folder: %s" % args.input_file)
        # imglist = glob.glob(args.input_file + '/*.' + args.ext)
        # inputs =[caffe.io.load_image(im_f) - PIXEL_MEANS
                 # for im_f in imglist]
    # elif osp.splitext(args.input_file)[1] == '.txt':
        # imglist = [imgPath + l.split('\t')[0] for l in open(gtFile,'r').readlines()]
        # inputs =[caffe.io.load_image(im_f) - PIXEL_MEANS
                 # for im_f in imglist]
    # else: # image
        # print("Loading file: %s" % args.input_file)
        # imglist = [args.input_file]
        # inputs = [caffe.io.load_image(args.input_file) - PIXEL_MEANS]        
        # # img = caffe.io.load_image(args.input_file)
        # # transformed_image = classifier.transformer.preprocess('data', img)
        # # print transformed_image.shape
        # #plt.imshow(img)
        # #plt.show()
        # # print img.shape
        # # print img

    # print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    # predictions = classifier.predict(inputs, not args.center_only)
    # print predictions
    conf_thrsh = 0.01
    overallAccuracy = 0
    vis = False
    saveVis = False
    all_scores = []
    all_img_names = []
    for ix, im_f in enumerate(imglist):
        imgFile = imgPath + im_f
        print imgFile
        all_img_names.append(im_f)
        inputs = [caffe.io.load_image(imgFile) - PIXEL_MEANS]
        predictions = classifier.predict(inputs, not args.center_only)
        imgName = osp.basename(im_f)
        imgLabelStr = name_label_map[imgName] if imgName in name_label_map else 'None'
        gtLabels = imgLabelStr.split(',')
        gtNum = len(gtLabels)
        scores = predictions[0]
        # print scores
        all_scores.append(scores)
        cls_ids = np.where(scores >= conf_thrsh)[0]
        sorted_ind = np.argsort(-scores)
        sorted_scores = -np.sort(-scores)
        top_sorted_ind = sorted_ind[:gtNum]
        pred_cls = [classes[id] for id in cls_ids]
        top_pred_cls = [classes[id] for id in top_sorted_ind]
        imgAcc = 0
        for gl in gtLabels:
            if gl in top_pred_cls:
                imgAcc += 1
        imgAcc /= float(gtNum)
        predStr = 'Predict:'
        gtStr =  "GT Labels: {}".format(imgLabelStr)
        print gtStr
        for id in cls_ids:
            str = "{}:{:.3f}".format(classes[id], scores[id])
            predStr += str + '||'
            print str
        if vis:
            plt.cla()
            img = caffe.io.load_image(imgFile)
            plt.imshow(img)
            plt.axis('off')
            title = "{}\n{}\nImage Accuracy = {:.3f}".format(gtStr, predStr, imgAcc)
            plt.title(title)
            vis_image_name = osp.join(visFolder, imgName + '.png')
            if saveVis and vis_image_name is not None:
                plt.savefig(vis_image_name, bbox_inches='tight',pad_inches = 0)
            if len(imglist) < 10:
                plt.show()
        overallAccuracy += imgAcc
    print("Done in %.2f s." % (time.time() - start))
    overallAccuracy /= len(imglist)
    print "Overall Multilabel Accuracy = {:.3f}".format(overallAccuracy)
    

    # Save
    if output_pred_file is not None:
        print("Saving results into %s" % output_pred_file)
        str_data = np.char.mod("%1.4f", all_scores)
        rows = np.array(all_img_names)[:, np.newaxis]
        np.savetxt(output_pred_file + '.txt', np.hstack((rows, str_data)), fmt='%s')
        np.save(output_pred_file + '.npy',all_scores)


if __name__ == '__main__':
    main(sys.argv)

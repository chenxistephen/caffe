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
from glob import glob

imgPath = '../VisualIntent/Images/'
labelFile = '../VisualIntent/taxonomy/visual_intent_30_labels.txt'
gtFile = '../VisualIntent/Annotations_multiclass/val.txt'
imgsetFile = '../VisualIntent/ImageSets/val.txt'
classes = [l.rstrip() for l in open(labelFile,'r').readlines()]


def eval_multilabel_cls(all_scores):
    gtLabels = parseGT(gtFile, classes)
    np.savetxt('tmp.gt',gtLabels,fmt="%d")
    outputDir = '../outputs/pred_results/'
    outFile = osp.join(outputFolder, 'map_' + modelName + '.tsv')
    use_07_metric = False
    
    imglist = [l.rstrip() for l in open(imgsetFile,'r').readlines()]
    # all_scores = np.load(resFile)
    nd = all_scores.shape[0]
    numClass = all_scores.shape[1]
    if top_k < 10:
        for ix in range(len(all_scores)):
            img_score = all_scores[ix,:]
            img_sorted_ind = np.argsort(-img_score)
            img_top_ind = img_sorted_ind[:top_k]
            mask = np.zeros(numClass)
            mask[img_top_ind] = 1
            all_scores[ix,:] = np.multiply(img_score, mask)
        # print img_score
        # print mask
        # print all_scores[ix,:]
    all_rec = []
    all_prec = []
    all_ap = []
    all_npos = []

    for c, cls in enumerate(classes):
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        cls_gts = gtLabels[:,c]
        npos = len(np.where(cls_gts == 1)[0])
        all_npos.append(npos)
        cls_scores = all_scores[:,c]
        sorted_ind = np.argsort(-cls_scores)
        sorted_scores = cls_scores[sorted_ind]
        # sorted_preds = (sorted_scores >= conf_thresh).astype(int)
        sorted_gts = cls_gts[sorted_ind]
        # tpids = np.where( (sorted_preds == 1) & (sorted_gts == 1))[0]
        # fpids = np.where((sorted_preds == 1) & (sorted_gts == 0))[0]
        # print "npos = {}, tpNum = {}, fpNum = {}".format(npos, len(tpids), len(fpids))
        # tp[tpids] = 1
        # fp[fpids] = 1
        for d in range(nd):
            s = sorted_scores[d]
            # if cls_gts[sorted_ind[d]] == 1: # GT has the class
            if s > conf_thresh:
                if sorted_gts[d] == 1: # GT has the class                
                    tp[d] = 1
                else:
                    fp[d] = 1
        tpNum = len(np.where(tp==1)[0])
        fpNum = len(np.where(fp==1)[0])

        
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # print tp
        
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        
        print "nposGT = {}, nPred = {}, tpNum = {}, fpNum = {}".format(npos, int(max(tp+fp)), tpNum, fpNum)
        #rec, prec, ap
        all_rec.append(rec)
        all_prec.append(prec)
        all_ap.append(ap)
        print "{}: ap = {}".format(cls, ap)
    map = np.mean(all_ap)
    all_npos = all_npos / np.sum(all_npos).astype(float)
    # print np.sum(all_npos)
    # print "weights = {}".format(all_npos)
    weighted_map = np.sum(np.multiply(all_ap, all_npos))
    map_ind = np.argsort(-np.array(all_ap))
    with open(outFile,'wb') as f:
        for i in map_ind:
            f.write("{}\t{:.3f}\n".format(classes[i], all_ap[i]))
        f.write("mAP = {:.3f}\n".format(map))
        f.write("Weighted AP = {:.3f}\n".format(weighted_map))
        
    print "mAP = {:.3f}".format(map)
    print "Weighted AP = {:.3f}".format(weighted_map)


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)


    input_scale = 1.0 #0.017 # 1.0
    raw_scale = 255.0 # [1.0, 255.0]
    vis = False
    saveVis = True

    
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
        default='../outputs',
        help="Output npy filename."
    )
    parser.add_argument(
        "--modelDir",
        default='../../trained_models/VisualIntent/caffenet/softmax_50k/',
        help="Dir of the trained model"
    )
    parser.add_argument(
        "--output_pred_file",
        default=None,
        help="Output npy filename."
    )
    # Optional arguments.
    # parser.add_argument(
        # "--model_def",
        # default=os.path.join(pycaffe_dir,
                # modelDir, "deploy.prototxt"),
        # help="Model definition file."
    # )
    # parser.add_argument(
        # "--pretrained_model",
        # default=os.path.join(pycaffe_dir,
                # modelDir, "caffenet_softmax_iter_50000.caffemodel"),
        # help="Trained model weights file."
    # )
    parser.add_argument(
        "--gpu",
        default=True,
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0, #255.0,
        help="Multiply raw input by this scale before preprocessing."
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
        default=input_scale,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=raw_scale, #255.0,
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
    deploy_prototxt = osp.join(args.modelDir, 'deploy.prototxt')
    trained_model = glob(osp.join(args.modelDir, '*.caffemodel'))[0]
    
    # PIXEL_MEANS = np.array([[[128.223/255.0,135.409/255.0,142.853/255.0]]])
    PIXEL_MEANS = np.array([[[103.94/255.0, 116.78/255.0, 123.68/255.0]]])
    # PIXEL_MEANS = np.array([[[103.94, 116.78, 123.68]]])
    
    
    print PIXEL_MEANS
    
    output_dir = args.output_dir
    visDir = osp.join(output_dir, 'visualization')
    resDir = osp.join(output_dir, 'pred_results')
    modelName = osp.splitext(osp.basename(trained_model))[0]
    print modelName
    visFolder = osp.join(visDir, modelName)
    outresFolder = osp.join(resDir, modelName)
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
        caffe.set_device(args.device_id)
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")
        

    # Make classifier.
    classifier = caffe.Classifier(deploy_prototxt, trained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)
    print "device_id = {}".format(args.device_id)
    print "deploy_prototxt = {}".format(deploy_prototxt)
    print "trained_model = {}".format(trained_model)
    print "args.input_scale = {}".format(args.input_scale)
    print "args.raw_scale = {}".format(args.raw_scale)

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
    conf_thrsh = 0.1 #01
    overallAccuracy = 0

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
        print "Image Accuracy = {:.3f}".format(imgAcc)
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

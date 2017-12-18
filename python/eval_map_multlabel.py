import os
import os.path as osp
import sys
import numpy as np


def parseGT(annofile, classes=None):
    lines = open(annofile,'r').readlines()
    imgNum = len(lines)
    classNum = len(classes)
    cls_id_map = dict(zip(classes, range(classNum)))
    gtLabels =  np.zeros((imgNum, classNum))
    for ix, line in enumerate(lines):
        labels = line.rstrip().split('\t')[-1].split(',')
        for c in labels:
            gtLabels[ix, cls_id_map[c]] = 1
    return gtLabels
    
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


conf_thresh = 0 #0.9
top_k = 5000

if __name__ == '__main__':
    resFile = sys.argv[1] if len(sys.argv) > 1 else '../visualizations/pred_results/caffenet_softmax_iter_50000.npy'
    imglistFile = '../VisualIntent/ImageSets/val.txt'
    labelFile = '../VisualIntent/taxonomy/visual_intent_30_labels.txt'
    annoFile = '../VisualIntent/Annotations_multiclass/val.txt'
    classes = [l.rstrip() for l in open(labelFile,'r').readlines()]
    gtLabels = parseGT(annoFile, classes)
    np.savetxt('tmp.gt',gtLabels,fmt="%d")
    outFile = '../visualizations/pred_results/map_caffenet_softmax_iter_50000.tsv'
    use_07_metric = False

    imglist = [l.rstrip() for l in open(imglistFile,'r').readlines()]

    all_scores = np.load(resFile)
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
    

import os
import os.path as osp
import sys
import numpy as np

saveVisPath = './visualization'
#print class_id_dict
DATASET_LIST = {'HomeFurniture', 'FashionV2', 'FashionV2-CatMap', 'BingMeasurement', 'VisualIntent'}
MERGED_LIST = {'FashionV2-CatMap'}
IGNORED_CLSERROR_LIST = {'__deleted__', '__background__'}
DATASET='FashionV2' # {'HomeFurniture', 'FashionV2', 'BingMeasurement'}
annopath = './data/VisualIntent/Annotations_multiclass/val.txt'
imagesetfile = './data/VisualIntent/ImageSets/val.txt'
classLabelFile = './data/VisualIntent/taxonomy/visual_intent_30_labels.txt'
imgPath = './data/VisualIntent/Images/'
cachedir = './data/VisualIntent/annotations_cache/'
clswiseThrshFile = None
class_threshold_dict = None
classes = open(classLabelFile,'r').readlines()
classes = [c.rstrip() for c in classes]
classes = ['__background__'] + classes
class_id_dict = dict([(name, ix) for ix, name in enumerate(classes)])

classNum = len(classes)
allClassIds = range(classNum)

def setPaths(DATASET, numClass=0):
    print "Setting Dataset = {} Paths!".format(DATASET)
    global annopath
    global imagesetfile
    global classLabelFile
    global clswiseThrshFile
    global imgPath
    global cachedir
    global classes
    global class_id_dict
    global classNum
    global allClassIds
    global class_threshold_dict
    if DATASET == 'HomeFurniture':
        annopath = './data/HomeFurniture/Annotations_58/val.txt'
        imagesetfile = './data/HomeFurniture/ImageSets_58/val.txt'
        classLabelFile = './data/HomeFurniture/taxonomy/furniture_58_labels.txt'
        imgPath = './data/HomeFurniture/Images/'
        cachedir = './data/HomeFurniture/annotations_cache/'
    elif DATASET == 'FashionV2':
        annopath = './data/FashionV2/Annotations_multiclass/val.txt'
        imagesetfile = './data/FashionV2/ImageSets/val.txt'
        classLabelFile = './data/FashionV2/taxonomy/fashion_82_labels.txt'
        clswiseThrshFile = './data/FashionV2/thresholds/fashion_82_ProdThrsh_V1.txt'
        imgPath = './data/FashionV2/Images/'
        cachedir = './data/FashionV2/annotations_cache/'
    elif DATASET == 'FashionV2-CatMap':
        annopath = './data/FashionV2/Annotations_multiclass/val_{}.txt'.format(numClass)
        imagesetfile = './data/FashionV2/ImageSets/val.txt'
        imgPath = './data/FashionV2/Images/'
        cachedir = './data/FashionV2/annotations_cache/'
        classLabelFile = './data/FashionV2/taxonomy/FashionV2_refind_category_{}.txt'.format(numClass)
    elif DATASET == 'VisualIntent':
        annopath = './data/VisualIntent/Annotations_multiclass/val.txt'
        imagesetfile = './data/VisualIntent/ImageSets/val.txt'
        classLabelFile = './data/VisualIntent/taxonomy/visual_intent_30_labels.txt'
        imgPath = './data/VisualIntent/Images/'
        cachedir = './data/VisualIntent/annotations_cache/'
    elif DATASET == 'BingMeasurement':
        annopath = './data/BingMeasurement/Annotations_multiclass/val.txt'
        imagesetfile = './data/BingMeasurement/ImageSets/val.txt'
        classLabelFile = './data/BingMeasurement/taxonomy/fashion_82_labels.txt'
        imgPath = './data/BingMeasurement/Images/'
        cachedir = './data/BingMeasurement/annotations_cache/'
    class_threshold_dict = {l.split('\t')[0]:float(l.rstrip().split('\t')[1]) for l in open(clswiseThrshFile,'r').readlines()} if clswiseThrshFile is not None else {}
    
    classes = open(classLabelFile,'r').readlines()
    classes = [c.rstrip() for c in classes]
    # classes = ['__background__'] + classes
    class_id_dict = dict([(name, ix) for ix, name in enumerate(classes)])
    print "len(classes) = {}, classes = {}".format(len(classes), classes)
    classNum = len(classes)
    allClassIds = range(classNum)
bgid = 0
boxlen = 6
eps = sys.float_info.epsilon

def vis_det_error(im, detbox, detclsIdx, detscore, imgGT_all, jmaxIds, vis_image_name=None, title=''):
    if osp.exists(vis_image_name):
        #print "{} Exits. Continue!".format(vis_image_name)
        return
    detcolor='r'
    gtcolor='g'
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    plt.imshow(im)
    plt.axis('off')
    plt.title(title)
    ax = plt.gca()
    BBGT_all = imgGT_all['bbox'].astype(float)
    gtboxNum =  BBGT_all.shape[0]
    for jmax in jmaxIds:
        if jmax >= 0:
        #for gid in range(gtboxNum):
            #print box
            gid = jmax
            categ = imgGT_all['categ'][gid]
            bbox = BBGT_all[gid,:]
            ecolor = 'm' if gid == jmax else gtcolor
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=ecolor, linestyle='--', linewidth=3)
            )
            ax.text(bbox[0], bbox[3], categ, bbox={'facecolor': ecolor, 'alpha': 0.5})
    
    if len(detbox) > 0:
        bbox = detbox
        categ = detclsIdx
        score = detscore
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=detcolor, linewidth=3)
        )
        ax.text(bbox[0], bbox[1], '{}:{:.2f}'.format(classes[categ],score), bbox={'facecolor': detcolor, 'alpha': 0.5})
    if vis_image_name is not None:
        plt.savefig(vis_image_name, bbox_inches='tight',pad_inches = 0)
    else:
        plt.show()


def _get_voc_results_file_template(voc_res_path):
    return os.path.join(voc_res_path, 'comp4_det_test_{:s}.txt')

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
    
thresh = 0.05
def read_voc_results_files(voc_res_path):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    num_images = len(imagenames)
    num_classes = len(classes)
    all_boxes = [[[] for _ in xrange(num_images)]
        for _ in xrange(num_classes)]
    reorder = [2,3,4,5,1]
    for cls_ind, cls in enumerate(classes):
        if cls == '__background__':
            continue
        print 'Reading {} VOC results file: {}/{}'.format(cls, cls_ind, len(classes))
        filename = _get_voc_results_file_template(voc_res_path).format(cls)
        rlines = open(filename,'r').readlines()
        resNames = [l.split()[0] for l in rlines]
        print "len(resNames) = {}".format(len(resNames))
        for imgName in set(resNames):
            imgIdx = imagenames.index(imgName)
            #print "imgName = {}, imgIdx = {}".format(imgName, imgIdx)
            imgLineIds = [ix for ix, name in enumerate(resNames) if name == imgName]
            imgClsDet = []
            for lid in imgLineIds:
                #print rlines[lid]
                bbox = rlines[lid].rstrip().split()
                bbox = [float(bbox[i]) for i in reorder]
                if bbox[-1] > thresh:
                    #print bbox
                    imgClsDet.append(bbox)
            #print imgClsDet
            imgClsDet = np.array(imgClsDet)
            #print imgClsDet.shape
            all_boxes[cls_ind][imgIdx] = np.array(imgClsDet)
            #if all_boxes[cls_ind][imgIdx].shape[0] > 0:
            #    print all_boxes[cls_ind][imgIdx]
    return all_boxes 
    
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


conf_thrsh = 0.8 #0.9
top_k = 5000

if __name__ == '__main__':
    if len(sys.argv) > 1: #input CaffeProc detection file format
        vocDetFile = sys.argv[1]
    else:
        sys.exit("Usage: python .\src\caffeProc_error_analysis_frcnnout.py <Caffe Processor output file> [optional: vis]")
     
    
    datasetFolder =  os.path.splitext(vocDetFile)[0].split('results\\')[1]
    #print "datasetFolder.split('_')[0] = {} =================".format(datasetFolder.split('_')[0])
    if datasetFolder.split('_')[0] in DATASET_LIST:
        DATASET = datasetFolder.split('_')[0]
        numClass = int(datasetFolder.split('\\')[0].split('_')[-1])
        print "DATASET = {}, numClass = {} =================.".format(DATASET, numClass)
        setPaths(DATASET, numClass)
    print "imagesetfile = {}".format(imagesetfile)
    if os.path.isdir(vocDetFile):
        model_name =  os.path.splitext(vocDetFile)[0].split('\\')[-1]
        if len(model_name) == 0:
            model_name =  os.path.splitext(vocDetFile)[0].split('\\')[-2]
        print "model_name = {} ================= ".format(model_name)
        voc_res_path = vocDetFile
        # all_boxes = read_voc_results_files(vocDetFile)
        # all_boxes = np.array(all_boxes)
        # det_file = os.path.join(vocDetFile, 'all_boxes.pkl')
        # with open(det_file, 'wb') as f:
            # cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    # elif os.path.exists(vocDetFile):
        # model_name =  os.path.splitext(vocDetFile)[0].split('\\')[-2]
        # print "model_name = {} ================= ".format(model_name)
        # all_boxes = cPickle.load(open(vocDetFile,'rb'))
        # all_boxes = np.array(all_boxes)


    classIds = xrange(1, len(classes))

    # resFile = sys.argv[1] if len(sys.argv) > 1 else '../visualizations/pred_results/caffenet_softmax_iter_50000.npy'
    # imagesetfile = '../VisualIntent/ImageSets/val.txt'
    # classLabelFile = '../VisualIntent/taxonomy/visual_intent_30_labels.txt'
    # annopath = '../VisualIntent/Annotations_multiclass/val.txt'
    # classes = [l.rstrip() for l in open(classLabelFile,'r').readlines()]
    print annopath
    gtLabels = parseGT(annopath, classes)
    # print "gtLabels.shape = {}".format(gtLabels.shape)
    # np.savetxt('tmp.gt',gtLabels,fmt="%d")
    outFile = './pred_results/map_caffenet_softmax_iter_50000.tsv'
    use_07_metric = False

    imglist = [l.rstrip() for l in open(imagesetfile,'r').readlines()]
    imageName_dict = dict([(name, ix) for ix, name in enumerate(imglist)])
    nd = len(imglist)

    # all_scores = np.load(resFile)
    # nd = all_scores.shape[0]
    # numClass = all_scores.shape[1]
    # if top_k < 10:
        # for ix in range(len(all_scores)):
            # img_score = all_scores[ix,:]
            # img_sorted_ind = np.argsort(-img_score)
            # img_top_ind = img_sorted_ind[:top_k]
            # mask = np.zeros(numClass)
            # mask[img_top_ind] = 1
            # all_scores[ix,:] = np.multiply(img_score, mask)
        # # print img_score
        # # print mask
        # # print all_scores[ix,:]
        
        


    all_rec = []
    all_prec = []
    all_ap = []
    all_npos = []

    for c, cls in enumerate(classes):
        if cls == '__background__':
            continue
        print "{}:{}".format(c, cls)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        cls_scores = np.zeros(nd)
        
        cls_gts = gtLabels[:,c]
        npos = len(np.where(cls_gts == 1)[0])
        all_npos.append(npos)

        
        # Read detection results
        det_file = _get_voc_results_file_template(voc_res_path).format(cls)
        with open(det_file, 'r') as f:
            rlines = f.readlines()
        splitlines = [x.strip().split(' ') for x in rlines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        for ix, s in enumerate(confidence):
            imgIdx = imageName_dict[image_ids[ix]]
            cls_scores[imgIdx] = max(cls_scores[imgIdx], s)
        
        # for ix, imgName in enumerate(imglist):
            # # print "imgName = {}, len(imgids) = {}".format(imgName, len(imgids))
            # img_confidence = [confidence[ii] for ii, n in enumerate(image_ids) if n == imgName]
            # if len(img_confidence) > 0:
                # cls_scores[ix] = max(img_confidence)
        
            
        
        # sort by confidence
        # print "len(cls_scores) = {}".format(len(cls_scores))
        sorted_ind = np.argsort(-cls_scores)
        sorted_scores = cls_scores[sorted_ind]
        imglist_sorted = [imglist[x] for x in sorted_ind]
        # print sorted_ind
        # print sorted_scores
        sorted_gts = cls_gts[sorted_ind]
        # print len(sorted_scores)
        
       

        
        for d in range(nd):
            imgName = imglist_sorted[d]
            # imgIdx = imageName_dict[imgName]
            # print "imgName = {}, imgIdx = {}".format(imgName, imgIdx)
            s = sorted_scores[d]
            # if cls_gts[sorted_ind[d]] == 1: # GT has the class
            if s > conf_thrsh:
                if sorted_gts[d] == 1:#cls_gts[imgIdx] == 1: # GT has the class                
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
        
        print "Class = {}: nposGT = {}, nPred = {}, tpNum = {}, fpNum = {}".format(cls, npos, int(max(tp+fp)), tpNum, fpNum)
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
    print "mAP = {:.3f}".format(map)
    print "Weighted AP = {:.3f}".format(weighted_map)
    with open(outFile,'wb') as f:
        for i in map_ind:
            f.write("{}\t{:.3f}\n".format(classes[i], all_ap[i]))
        f.write("mAP = {:.3f}\n".format(map))
        f.write("Weighted AP = {:.3f}\n".format(weighted_map))
        

    

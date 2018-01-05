
"""

"""
import argparse
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from CaffeNetBuilder import CaffeNetBuilder

def MakeGooglenet(nb, name, num_classes):
    conv1 = nb.ConvFactory(["data"], 64, kernel=(7, 7), stride=(2,2), pad=(3, 3), name="%s_conv1"%(name))
    pool1 = nb.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max", name="%s_pool1"%(name))
    lrn1 = nb.LRN(pool1, local_size=5, alpha=0.0001, beta=0.75, name="%s_lrn1"%(name))
    conv2 = nb.ConvFactory(lrn1, 64, kernel=(1, 1), stride=(1,1), name="%s_conv2"%(name))
    conv3 = nb.ConvFactory(conv2, 192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="%s_conv3"%(name))
    lrn3 = nb.LRN(conv3, local_size=5, alpha=0.0001, beta=0.75, name="%s_lrn3"%(name))
    pool3 = nb.Pooling(lrn3, kernel=(3, 3), stride=(2, 2), pool_type="max", name="%s_pool3"%(name))

    in3a = nb.InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name="%s_in3a"%(name))
    in3b = nb.InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="%s_in3b"%(name))
    pool4 = nb.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max", name="%s_pool4"%(name))
    in4a = nb.InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name="%s_in4a"%(name))

    loss1_pool = nb.Pooling(in4a, kernel=(5, 5), stride=(3, 3), pool_type="avg", name="%s_loss1_pool"%(name))
    loss1_conv = nb.ConvFactory(loss1_pool, 128, kernel=(1, 1), stride=(1, 1), name="%s_loss1_conv"%(name))
    loss1_fc = nb.FullyConnected(loss1_conv, num_hidden=1024, name="%s_loss1_fc"%(name))
    loss1_fc_relu = nb.ReLU(loss1_fc, name="%s_loss1_fc_relu"%(name)) 
    loss1_fc_drop = nb.Dropout(loss1_fc_relu, dropout_ratio=0.7, name="%s_loss1_fc_drop"%(name))
    loss1_classifier = nb.FullyConnected(loss1_fc_drop, num_hidden=num_classes, name="%s_loss1_classifier"%(name)) 
    loss1 = nb.SoftMaxWithLoss(loss1_classifier, label="label", loss_weight=0.3, name='%s_loss1'%(name))
    accu1_top1 = nb.Accuracy(loss1_classifier, label="label", top_k=1, name='%s_accu1_top1'%(name))
    accu1_top5 = nb.Accuracy(loss1_classifier, label="label", top_k=5, name='%s_accu1_top5'%(name))
    
    in4b = nb.InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="%s_in4b"%(name))
    in4c = nb.InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="%s_in4c"%(name))
    in4d = nb.InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="%s_in4d"%(name))
    
    loss2_pool = nb.Pooling(in4d, kernel=(5, 5), stride=(3, 3), pool_type="avg", name="%s_loss2_pool"%(name))
    loss2_conv = nb.ConvFactory(loss2_pool, 128, kernel=(1, 1), stride=(1, 1), name="%s_loss2_conv"%(name))
    loss2_fc = nb.FullyConnected(loss2_conv, num_hidden=1024, name="%s_loss2_fc"%(name))
    loss2_fc_relu = nb.ReLU(loss2_fc, name="%s_loss2_fc_relu"%(name)) 
    loss2_fc_drop = nb.Dropout(loss2_fc_relu, dropout_ratio=0.7, name="%s_loss2_fc_drop"%(name))
    loss2_classifier = nb.FullyConnected(loss2_fc_drop, num_hidden=num_classes, name="%s_loss2_classifier"%(name)) 
    loss2 = nb.SoftMaxWithLoss(loss2_classifier, label="label", loss_weight=0.3, name='%s_loss2'%(name))
    accu2_top1 = nb.Accuracy(loss2_classifier, label="label", top_k=1, name='%s_accu2_top1'%(name))
    accu2_top5 = nb.Accuracy(loss2_classifier, label="label", top_k=5, name='%s_accu2_top5'%(name))
    
    in4e = nb.InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="%s_in4e"%(name))
    pool5 = nb.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pool_type="max", name="%s_pool5"%(name))
    in5a = nb.InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name="%s_in5a"%(name))
    in5b = nb.InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="%s_in5b"%(name))
    
    loss3_pool = nb.Pooling(in5b, kernel=(7, 7), stride=(1, 1), pool_type="avg", name = "%s_loss3_pool"%(name))    
    loss3_drop = nb.Dropout(loss3_pool, dropout_ratio=0.4, name="%s_loss3_drop"%(name))
    loss3_classifier = nb.FullyConnected(loss3_drop, num_hidden=num_classes, name="%s_loss3_classifier"%(name))    
    loss3 = nb.SoftMaxWithLoss(loss3_classifier, label="label", name='%s_loss3'%(name))
    accu3_top1 = nb.Accuracy(loss3_classifier, label="label", top_k=1, name='%s_accu3_top1'%(name))
    accu3_top5 = nb.Accuracy(loss3_classifier, label="label", top_k=5, name='%s_accu3_top5'%(name))
    
    nb.SetLayerTrainParam(layer_type="Convolution", weight_param=caffe_pb2.ParamSpec(lr_mult=1, decay_mult=1), bias_param=caffe_pb2.ParamSpec(lr_mult=2, decay_mult=0))
    nb.SetLayerInitializer(layer_type="Convolution", weight_filler = caffe_pb2.FillerParameter(type="xavier"), bias_filler = caffe_pb2.FillerParameter(type="constant", value=0.2))
    
    nb.SetLayerTrainParam(layer_type="InnerProduct", weight_param=caffe_pb2.ParamSpec(lr_mult=1, decay_mult=1), bias_param=caffe_pb2.ParamSpec(lr_mult=2, decay_mult=0))
    nb.SetLayerInitializer(layer_name=loss1_fc[-1].name, weight_filler = caffe_pb2.FillerParameter(type="xavier"), bias_filler = caffe_pb2.FillerParameter(type="constant", value=0.2)) 
    nb.SetLayerInitializer(layer_name=loss1_classifier[-1].name, weight_filler = caffe_pb2.FillerParameter(type="xavier"), bias_filler = caffe_pb2.FillerParameter(type="constant", value=0))
    nb.SetLayerInitializer(layer_name=loss2_fc[-1].name, weight_filler = caffe_pb2.FillerParameter(type="xavier"), bias_filler = caffe_pb2.FillerParameter(type="constant", value=0.2))
    nb.SetLayerInitializer(layer_name=loss2_classifier[-1].name, weight_filler = caffe_pb2.FillerParameter(type="xavier"), bias_filler = caffe_pb2.FillerParameter(type="constant", value=0))
    nb.SetLayerInitializer(layer_name=loss3_classifier[-1].name, weight_filler = caffe_pb2.FillerParameter(type="xavier"), bias_filler = caffe_pb2.FillerParameter(type="constant", value=0))
    
    return nb

def main():
    parser = argparse.ArgumentParser(description='Output a network graph')
    parser.add_argument('output_prototxt_file',
                        help='Output image file')
    args = parser.parse_args()
    
    net = caffe_pb2.NetParameter()
    net.name = "GoogleNet"
    nb = CaffeNetBuilder(net)
    MakeGooglenet(nb, 'gl', 1000)

    with open(args.output_prototxt_file, 'w') as of:
        of.write(text_format.MessageToString(net, as_utf8=True))


if __name__ == '__main__':
    main()

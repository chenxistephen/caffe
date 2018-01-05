from caffe.proto import caffe_pb2

class CaffeNetBuilder(object):
    def __init__(self, net):
        self.net = net

    def __GetTopData(self, data):
        if(data is None):
            raise Exception("none data input")
        elif(type(data)==caffe_pb2.LayerParameter):
            return data.top
        elif(type(data)==str or type(data)==unicode):
            return data
        elif(type(data) == type([])):
            if(type(data[-1])==str or type(data)==unicode):
                return [data[-1]]
            else:
                return data[-1].top
        else:
            return data[-1].top
    
    @staticmethod
    def __SetConvKernelParam(param, kernel, stride, pad):
        if(kernel[0] == kernel[1]):
            param.kernel_size.extend([kernel[0]])
        else:
            param.kernel_w = kernel[0]
            param.kernel_h = kernel[1]
            
        if(stride[0] == stride[1]):
            param.stride.extend([stride[0]])
        else:
            param.stride_w = stride[0]
            param.stride_h = stride[1]
            
        if(pad[0] == pad[1]):
            param.pad.extend([pad[0]])
        else:
            param.pad_w = pad[0]
            param.pad_h = pad[1]
            
    @staticmethod
    def __SetPoolKernelParam(param, kernel, stride, pad):
        if(kernel[0] == kernel[1]):
            param.kernel_size = kernel[0]
        else:
            param.kernel_w = kernel[0]
            param.kernel_h = kernel[1]
            
        if(stride[0] == stride[1]):
            param.stride = stride[0]
        else:
            param.stride_w = stride[0]
            param.stride_h = stride[1]
            
        if(pad[0] == pad[1]):
            param.pad = pad[0]
        else:
            param.pad_w = pad[0]
            param.pad_h = pad[1]       
            
    def ConvFactory(self, data, num_filter, kernel, stride=(1, 1), pad=(0, 0), name=None, suffix=''):
        data = self.__GetTopData(data)
        conv = caffe_pb2.LayerParameter(type="Convolution", name='%s%s_conv' %(name, suffix), bottom=data, top=['%s%s_conv' %(name, suffix)], \
                convolution_param=caffe_pb2.ConvolutionParameter(num_output=num_filter) \
            )
        CaffeNetBuilder.__SetConvKernelParam(conv.convolution_param, kernel, stride, pad)
        act = caffe_pb2.LayerParameter(type="ReLU", name='%s%s_relu' %(name, suffix), bottom=conv.top, top=conv.top)
        layers = [conv, act]
        self.net.layer.extend(layers)
        return layers
        
    def ConvBNFactory(self, data, num_filter, kernel, stride=(1, 1), pad=(0, 0), name=None, suffix=''):
        data = self.__GetTopData(data)
        conv = caffe_pb2.LayerParameter(type="Convolution", name='%s%s_conv' %(name, suffix), bottom=data, top=['%s%s_conv' %(name, suffix)], \
                convolution_param=caffe_pb2.ConvolutionParameter(num_output=num_filter) \
            )
        CaffeNetBuilder.__SetConvKernelParam(conv.convolution_param, kernel, stride, pad)
        batchNorm = caffe_pb2.LayerParameter(type="BatchNorm", name='%s%s_bn' %(name, suffix), bottom=conv.top, top=conv.top, batch_norm_param=caffe_pb2.BatchNormParameter())
        scale = caffe_pb2.LayerParameter(type="Scale", name='%s%s_scale' %(name, suffix), bottom=conv.top, top=conv.top, scale_param=caffe_pb2.ScaleParameter(bias_term=1))
        act = caffe_pb2.LayerParameter(type="ReLU", name='%s%s_relu' %(name, suffix), bottom=conv.top, top=conv.top)
        layers = [conv, batchNorm, scale, act]
        self.net.layer.extend(layers)
        return layers

    def ReLU(self, data, name=None):
        data = self.__GetTopData(data)
        relu = caffe_pb2.LayerParameter(type="ReLU", name=name, bottom=data, top=data)
        layers = [relu]
        self.net.layer.extend(layers)
        return layers
        
    def Pooling(self, data, kernel, stride=(1, 1), pad=(0, 0), pool_type="avg", name=None):
        pool_types_dic = {"MAX": caffe_pb2.PoolingParameter.MAX, "max": caffe_pb2.PoolingParameter.MAX, "AVE": caffe_pb2.PoolingParameter.AVE, "avg": caffe_pb2.PoolingParameter.AVE};
        data = self.__GetTopData(data)
        pooling = caffe_pb2.LayerParameter(bottom=data, type="Pooling", name=name, top=[name], \
                pooling_param = caffe_pb2.PoolingParameter(pool=pool_types_dic[pool_type]) \
            )
        CaffeNetBuilder.__SetPoolKernelParam(pooling.pooling_param, kernel, stride, pad)
        layers = [pooling]
        self.net.layer.extend(layers)
        return layers
        
    def LRN(self, data, local_size=5, alpha=1, beta=0.75, name=None):
        data = self.__GetTopData(data)
        lrn = caffe_pb2.LayerParameter(bottom=data, type="LRN", name=name, top=[name], \
                lrn_param = caffe_pb2.LRNParameter() \
            )
        lrn.lrn_param.local_size = local_size
        lrn.lrn_param.alpha = alpha
        lrn.lrn_param.beta = beta
        layers = [lrn]
        self.net.layer.extend(layers)
        return layers

    def Concat(self, data, name=None):
        bottoms = []
        for d in data:
            bottoms.extend(self.__GetTopData(d))
        concat = caffe_pb2.LayerParameter(bottom=bottoms, type="Concat", name=name,  top=[name])
        layers = [concat]
        self.net.layer.extend(layers)
        return layers
    
    def Dropout(self, data, dropout_ratio=0.5, name=None):
        data = self.__GetTopData(data)
        dropout = caffe_pb2.LayerParameter(bottom=data, type="Dropout", name=name,  top=data, \
                                      dropout_param = caffe_pb2.DropoutParameter(dropout_ratio=dropout_ratio)\
                                      )
        layers = [dropout]
        self.net.layer.extend(layers)
        return layers                            
    
    def FullyConnected(self, data, num_hidden, name=None, suffix=''):
        data = self.__GetTopData(data)
        fc = caffe_pb2.LayerParameter(bottom=data, type="InnerProduct", name=name,  top=[name], \
                                      inner_product_param = caffe_pb2.InnerProductParameter(num_output=num_hidden)\
                                      )
        layers = [fc]
        self.net.layer.extend(layers)
        return layers
        
    def FullyConnectedBNFactory(self, data, num_hidden, name=None, suffix=''):
        data = self.__GetTopData(data)
        fc = caffe_pb2.LayerParameter(bottom=data, type="InnerProduct", name=name,  top=[name], \
                                      inner_product_param = caffe_pb2.InnerProductParameter(num_output=num_hidden)\
                                      )
        batchNorm = caffe_pb2.LayerParameter(type="BatchNorm", name='%s%s_bn' %(name, suffix), bottom=fc.top, top=fc.top, batch_norm_param=caffe_pb2.BatchNormParameter())
        scale = caffe_pb2.LayerParameter(type="Scale", name='%s%s_scale' %(name, suffix), bottom=fc.top, top=fc.top, scale_param=caffe_pb2.ScaleParameter(bias_term=1))
        act = caffe_pb2.LayerParameter(type="ReLU", name='%s%s_relu' %(name, suffix), bottom=fc.top, top=fc.top)                              
        layers = [fc, batchNorm, scale, act]
        self.net.layer.extend(layers)
        return layers    
    
    def Accuracy(self, data, label, top_k=1, phase=1, name=None):
        data = self.__GetTopData(data)
        net_state_rule = [caffe_pb2.NetStateRule(phase=phase)]
        accuracy = caffe_pb2.LayerParameter(bottom=[data[0],label], type="Accuracy", name=name, top=[name], include=net_state_rule, \
                                            accuracy_param = caffe_pb2.AccuracyParameter(top_k=top_k)\
                                            )
        layers = [accuracy]
        self.net.layer.extend(layers)
        return layers
    
    def SoftMaxWithLoss(self, data, label, loss_weight=1, name=None):
        data = self.__GetTopData(data)       
        softmax = caffe_pb2.LayerParameter(bottom=[data[0],label], type="SoftmaxWithLoss", name=name,  top=[name], loss_weight=[loss_weight])
        layers = [softmax]
        self.net.layer.extend(layers)
        return layers

    def InceptionFactory(self, data, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name):
        #data = self.__GetTopData(data)
        # 1x1
        c1x1 = self.ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
        # 3x3 reduce + 3x3
        c3x3r = self.ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
        c3x3 = self.ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
        # double 3x3 reduce + double 3x3
        cd5x5r = self.ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), name=('%s_5x5' % name), suffix='_reduce')
        cd5x5 = self.ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), pad=(2, 2), name=('%s_5x5' % name))
        # pool + proj
        pooling = self.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_%s_pool' % (name, pool)))
        cproj = self.ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
        # concat
        concat = self.Concat([c1x1, c3x3, cd5x5, cproj], name='%s_concat' % name)
        layers = c1x1 + c3x3r + c3x3 + cd5x5r + cd5x5 + pooling + cproj + concat
        return layers
        
    def InceptionBNFactory(self, data, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name):
        #data = self.__GetTopData(data)
        # 1x1
        c1x1 = self.ConvBNFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
        # 3x3 reduce + 3x3
        c3x3r = self.ConvBNFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
        c3x3 = self.ConvBNFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
        # double 3x3 reduce + double 3x3
        cd5x5r = self.ConvBNFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), name=('%s_5x5' % name), suffix='_reduce')
        cd5x5 = self.ConvBNFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), pad=(2, 2), name=('%s_5x5' % name))
        # pool + proj
        pooling = self.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_%s_pool' % (name, pool)))
        cproj = self.ConvBNFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
        # concat
        concat = self.Concat([c1x1, c3x3, cd5x5, cproj], name='%s_concat' % name)
        layers = c1x1 + c3x3r + c3x3 + cd5x5r + cd5x5 + pooling + cproj + concat
        return layers
    
    def SetLayerTrainParam(self, layer_name=None, layer_type=None, weight_param=None, bias_param=None):
        if(layer_type!=None and layer_type!="Convolution" and layer_type!="InnerProduct"):
            raise Exception("not supported layer type for setting train param: %s"%(layer_type))
        if(weight_param==None or bias_param==None):
            raise Exception("params can't be None")
        for l in self.net.layer:
            if((layer_type != None and l.type == layer_type) or (layer_name != None and l.name == layer_name)):
                del l.param[:]
                l.param.extend([weight_param, bias_param])
                
    def SetLayerInitializer(self, layer_name=None, layer_type=None, weight_filler=None, bias_filler=None):
        for l in self.net.layer:
            if((layer_type != None and l.type == layer_type) or (layer_name != None and l.name == layer_name)):
                if(l.type=="Convolution"):
                    if(weight_filler != None):
                        l.convolution_param.weight_filler.CopyFrom(weight_filler)
                    if(bias_filler != None):
                        l.convolution_param.bias_filler.CopyFrom(bias_filler)
                elif(l.type=="InnerProduct"):
                    if(weight_filler != None):
                        l.inner_product_param.weight_filler.CopyFrom(weight_filler)
                    if(bias_filler != None):
                        l.inner_product_param.bias_filler.CopyFrom(bias_filler)
                else:
                    raise Exception("not supported layer type for setting initializer: %s"%(l.type))
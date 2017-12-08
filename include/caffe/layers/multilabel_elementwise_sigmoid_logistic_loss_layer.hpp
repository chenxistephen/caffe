#ifndef CAFFE_MULTILABEL_ELEMENTWISE_SIGMOID_LOGISTIC_LOSS_LAYER_HPP_
#define CAFFE_MULTILABEL_ELEMENTWISE_SIGMOID_LOGISTIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

  /**
  * @brief Computes the elementwise logistic loss for each class and
  *        combine all losses 
  */
  template <typename Dtype>
  class MultilabelElementwiseSigmoidLogisticLossLayer : public LossLayer<Dtype> {
  public:
    explicit MultilabelElementwiseSigmoidLogisticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
        sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
        sigmoid_output_(new Blob<Dtype>()) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "MultilabelElementwiseSigmoidLogisticLoss"; }

  protected:
    /// @copydoc MultilabelElementwiseSigmoidLogisticLoss
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    // The internal SigmoidLayer used to map predictions to probabilities.
    shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
    // sigmoid_output stores the output of the SigmoidLayer.
    shared_ptr<Blob<Dtype> > sigmoid_output_;
    // bottom vector holder to call the underlying SigmoidLayer::Forward
    vector<Blob<Dtype>*> sigmoid_bottom_vec_;
    // top vector holder to call the underlying SigmoidLayer::Forward
    vector<Blob<Dtype>*> sigmoid_top_vec_;

    // Positive Loss Weight
    Dtype positive_loss_weight_ = 1.0;
    Dtype negative_sample_ratio_ = 1.0;
    Blob<Dtype>* negative_samples_;

  };

}  // namespace caffe

#endif  // CAFFE_MULTILABEL_ELEMENTWISE_SIGMOID_LOGISTIC_LOSS_LAYER_HPP_

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multilabel_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // normalization factors
  vector<int> norm_factors_shape(4);
  norm_factors_shape[0] = bottom[0]->num();
  norm_factors_shape[1] = 1;
  norm_factors_shape[2] = 1;
  norm_factors_shape[3] = 1;
  norm_factors_ = new Blob<Dtype>(norm_factors_shape);
}

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
    << "Number of labels must match number of predictions; "
    << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
    << "label count (number of labels) must be N*H*W, "
    << "with integer values in {0, 1, ..., C-1}.";
}

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* norm_factors_data = norm_factors_->mutable_cpu_data();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  for (int i = 0; i < num; ++i) {
    norm_factors_data[i] = caffe_cpu_asum(dim, &label[i*dim]);
  }

  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < dim; k++) {
      if (static_cast<int>(label[i*dim + k]) != 0) {
        loss -= log(std::max(prob_data[i*dim + k], Dtype(FLT_MIN))) / Dtype(norm_factors_data[i]);
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* norm_factors_data = norm_factors_->cpu_data();
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    int dim = count / num;

    caffe_copy(prob_.count(), prob_data, bottom_diff);
    for (int i = 0; i < num; ++i) {
      for (int k = 0; k < dim; k++) {
        if (static_cast<int>(label[i*dim + k]) != 0) {
          bottom_diff[i * dim + k] -= 1.0 / Dtype(norm_factors_data[i]);
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / num;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultilabelSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(MultilabelSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MultilabelSoftmaxWithLoss);

}  // namespace caffe

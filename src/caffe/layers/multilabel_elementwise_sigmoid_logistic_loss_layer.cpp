#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <boost/bind/bind.hpp>
#include <stdlib.h>
#include <time.h>

#include "caffe/layers/multilabel_elementwise_sigmoid_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  
template <typename Dtype>
  void MultilabelElementwiseSigmoidLogisticLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[0]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(sigmoid_output_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

    positive_loss_weight_ = this->layer_param_.elementwise_sigmoid_logistic_loss_parameter().positive_loss_weight();
    negative_sample_ratio_ = this->layer_param_.elementwise_sigmoid_logistic_loss_parameter().negative_sample_ratio();
    vector<int> negative_samples_shape(4);
    negative_samples_shape[0] = bottom[0]->num();
    negative_samples_shape[1] = bottom[0]->count() / bottom[0]->num();
    negative_samples_shape[2] = 1;
    negative_samples_shape[3] = 1;
    negative_samples_ = new Blob<Dtype>(negative_samples_shape);
  }
  
  template <typename Dtype>
  void MultilabelElementwiseSigmoidLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
  }

  template <typename Dtype>
  void MultilabelElementwiseSigmoidLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the sigmoid outputs.
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    // Compute the loss (negative log likelihood)
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    //Dtype* positive_categories_count = positive_categories_->mutable_cpu_data();
    Dtype* negative_samples = negative_samples_->mutable_cpu_data();

    int count = bottom[0]->count();
    int num = bottom[0]->num();
    int dim = count / num;
    int r;

    Dtype loss = 0.0;
    for (int i = 0; i < num; ++i) {
      Dtype positive_loss = 0.0;
      Dtype negative_loss = 0.0;
      for (int l = 0; l < dim; ++l) {
        Dtype x = bottom_data[i * dim + l];
        if (bottom_label[i * dim + l] == 1) {
          positive_loss -= (x * (1 - (x >= 0)) - log(1 + exp(x - 2 * x * (x >= 0))));
          negative_samples[i * dim + l] = 1;
        }
        else {
          if (rand() % 100  <= negative_sample_ratio_ * 100) {
            negative_loss -= (x * (-(x >= 0)) - log(1 + exp(x - 2 * x * (x >= 0))));
            negative_samples[i * dim + l] = -1;
          }
          else {
            negative_samples[i * dim + l] = 0;
          }
        }
      }
      loss += positive_loss*positive_loss_weight_ + negative_loss;
    }
    top[0]->mutable_cpu_data()[0] = loss / count;
  }

  template <typename Dtype>
  void MultilabelElementwiseSigmoidLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* bottom_label = bottom[1]->cpu_data();
      const Dtype* negative_samples = negative_samples_->cpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      int count = bottom[0]->count();
      int num = bottom[0]->num();
	    int dim = count / num;
      const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
	    for (int i = 0; i < num; ++i) {
        for (int l = 0; l < dim; ++l) {
          if (negative_samples[i * dim + l] == 1) {
            bottom_diff[i * dim + l] = (sigmoid_output_data[i * dim + l] - (bottom_label[i * dim + l] == 1)) * positive_loss_weight_;
          }
          else if (negative_samples[i * dim + l] == 0) {
            bottom_diff[i * dim + l] = 0;
          }
          else {
            bottom_diff[i * dim + l] = (sigmoid_output_data[i * dim + l] - (bottom_label[i * dim + l] == 1));
          }
        }
	    }
      // Scale down gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
	    caffe_scal(count, loss_weight / count, bottom_diff);
    }
  }

#ifdef CPU_ONLY
  STUB_GPU(MultilabelElementwiseSigmoidLogisticLossLayer);
#endif

  INSTANTIATE_CLASS(MultilabelElementwiseSigmoidLogisticLossLayer);
  REGISTER_LAYER_CLASS(MultilabelElementwiseSigmoidLogisticLoss);

}  // namespace caffe

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/multilabel_elementwise_sigmoid_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void MultilabelElementwiseSigmoidLogisticLossBackwardGPU(const int nthreads,
    const Dtype* bottom_data, const Dtype* bottom_label, const Dtype* sigmoid_output_data, const Dtype* negative_samples, Dtype* bottom_diff,
    const int num, const int dim, const Dtype positive_loss_weight) {
	CUDA_KERNEL_LOOP(index, nthreads) {
	  const int i = index / dim;
	  const int l = index % dim;
      if (negative_samples[i * dim + l] == 1) {
        bottom_diff[i * dim + l] = (sigmoid_output_data[i * dim + l] - (bottom_label[i * dim + l] == 1)) * positive_loss_weight;
      }
      else if (negative_samples[i * dim + l] == 0) {
        bottom_diff[i * dim + l] = 0;
      }
      else {
        bottom_diff[i * dim + l] = sigmoid_output_data[i * dim + l] - (bottom_label[i * dim + l] == 1);
      }
    }
  }

  template <typename Dtype>
  void MultilabelElementwiseSigmoidLogisticLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      const Dtype* bottom_label = bottom[1]->gpu_data();
      const Dtype* negative_samples = negative_samples_->gpu_data();
      const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      int count = bottom[0]->count();
      int num = bottom[0]->num();
	  int dim = count / num;
	  int nthreads = count;

	  MultilabelElementwiseSigmoidLogisticLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS >> >(nthreads, bottom_data, bottom_label, sigmoid_output_data, negative_samples, bottom_diff, num, dim, positive_loss_weight_);

      // Scale down gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
	    caffe_gpu_scal(count, loss_weight / count, bottom_diff);
    }
  }

  INSTANTIATE_LAYER_GPU_BACKWARD(MultilabelElementwiseSigmoidLogisticLossLayer);


}  // namespace caffe
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multilabel_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void NormFactors(const int nthreads, const Dtype* label, Dtype* norm_factors_data,
           const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    norm_factors_data[index] = 0;
    for (int k = 0; k < dim; ++k) {
      norm_factors_data[index] += label[index*dim + k];
    }
  }
}

template <typename Dtype>
__global__ void MultilabelSoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss_data,
          Dtype* norm_factors_data, const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / dim;
    const int k = index % dim;
    const int label_value = static_cast<int>(label[index]);
    loss_data[index] = label_value == 0 ? 0 : -log(max(prob_data[i*dim + k], Dtype(FLT_MIN))) / Dtype(norm_factors_data[i]);
  }
}

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* norm_factors_data = norm_factors_->mutable_gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  
  NormFactors<Dtype> << <CAFFE_GET_BLOCKS(num),
    CAFFE_CUDA_NUM_THREADS >> >(num, label, norm_factors_data, dim);
  const int nthreads = count;
  MultilabelSoftmaxLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      norm_factors_data, dim);
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
__global__ void MultilabelSoftmaxLossBackwardGPU(const int nthreads,
          Dtype* bottom_diff, const Dtype* label,
          const Dtype* norm_factors_data, const int dim) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / dim;
    const int k = index % dim;
    const int label_value = static_cast<int>(label[index]);
    if (label_value != 0) {
      bottom_diff[i * dim + k] -= 1.0 / Dtype(norm_factors_data[i]);
    }
  }
}

template <typename Dtype>
void MultilabelSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const Dtype* norm_factors_data = norm_factors_->gpu_data();
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    int dim = count / num;
    const int nthreads = count;

    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    MultilabelSoftmaxLossBackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_diff, label, norm_factors_data, dim);

    Dtype loss_weight = top[0]->cpu_diff()[0] / num;
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(MultilabelSoftmaxWithLossLayer);

}  // namespace caffe

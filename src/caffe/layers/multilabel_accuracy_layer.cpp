#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multilabel_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultilabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void MultilabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultilabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  
  int num = bottom[0]->num();
  int dim = bottom[1]->count() / bottom[1]->num();

  for (int i = 0; i < num; ++i) {
    // Multi-label Accuracy

    std::vector<std::pair<Dtype, int> > score(dim);
    for (int j = 0; j < dim; j++) {
      score[j] = std::make_pair(bottom_data[i * dim + j], j);
    }
    sort(score.begin(), score.end());

    std::set<int> labels;
    for (int j = 0; j < dim; j++) {
      if (bottom_label[i * dim + j] > 0) labels.insert(j);
    }

    CHECK_GT(labels.size(), 0);
    int acc = 0;
    for (int j = dim - 1; j >= dim - (int)labels.size(); j--) {
      if (labels.count(score[j].second)) acc++;
    }

    accuracy += (Dtype)acc / labels.size();
  }

  top[0]->mutable_cpu_data()[0] = accuracy / num;
}

INSTANTIATE_CLASS(MultilabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultilabelAccuracy);

}  // namespace caffe

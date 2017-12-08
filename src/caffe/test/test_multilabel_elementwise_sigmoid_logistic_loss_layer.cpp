#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/multilabel_elementwise_sigmoid_logistic_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class MultilabelElementwiseSigmoidLogisticLossTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
	 MultilabelElementwiseSigmoidLogisticLossTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    PositiveUnitballFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int count = blob_bottom_label_->count();
    int num = blob_bottom_label_->num();
    int dim = count / num;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; ++j) {
        blob_bottom_label_->mutable_cpu_data()[i*dim + j] = 0.0;
      }
      blob_bottom_label_->mutable_cpu_data()[i*dim + caffe_rng_rand() % dim] = 1.0;
      blob_bottom_label_->mutable_cpu_data()[i*dim + caffe_rng_rand() % dim] = 1.0;
      blob_bottom_label_->mutable_cpu_data()[i*dim + caffe_rng_rand() % dim] = 1.0;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~MultilabelElementwiseSigmoidLogisticLossTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultilabelElementwiseSigmoidLogisticLossTest, TestDtypesAndDevices);

TYPED_TEST(MultilabelElementwiseSigmoidLogisticLossTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultilabelElementwiseSigmoidLogisticLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe

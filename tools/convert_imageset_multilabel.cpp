// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <iostream>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>


using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the image_datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in image_datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_bool(multilabel, false,
  "When this option is on, treat data as multilabel");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }
  
  CHECK((FLAGS_resize_height == 0 && FLAGS_resize_width == 0 && FLAGS_encoded) 
    || ((FLAGS_resize_height != 0 || FLAGS_resize_width != 0) && !FLAGS_encoded))
    << "Resizing is not supported when encoded is on. Do not set resized_width and resize_height";

  const string input_path = argv[1];
  const string db_path = argv[2];
  const string category_list_path = argv[3];

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const bool is_multilabel = FLAGS_multilabel;

  std::ifstream infile(input_path.c_str());

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB for image
  scoped_ptr<db::DB> image_db(db::GetDB(FLAGS_backend));
  image_db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> image_db_txn(image_db->NewTransaction());

  // Create new DB for multilabel
  scoped_ptr<db::DB> labels_db;
  scoped_ptr<db::Transaction> labels_db_txn;
  if (is_multilabel) {
    labels_db.reset(db::GetDB(FLAGS_backend));
    labels_db->Open(db_path + "/labels", db::NEW);
    labels_db_txn.reset(labels_db->NewTransaction());
  }
  
  string line;
  int count = 0;

  // Read category mapping file
  std::unordered_map<std::string, int> category_mapping;
  std::ifstream categoryFile(category_list_path.c_str());
  
  while (std::getline(categoryFile, line))
  {
    boost::trim_if(line, boost::is_any_of(" \n\r\t"));
    category_mapping.insert({ line, count });
    ++count;
  }
  int category_count = count;

  // Storing to db
  // std::string root_folder(argv[1]);
  Datum image_datum;
  std::vector<std::string> tuple;
  std::string labels, img_file;
  int data_size = 0, image_db_label;
  bool data_size_initialized = false;
  count = 0;
  int corrupted_image_count = 0;
  
  
  while (std::getline(infile, line))
  {
    boost::trim_if(line, boost::is_any_of(" \n\r\t"));
    boost::split(tuple, line, boost::is_any_of("\t"));
    CHECK_EQ(tuple.size(), 2) << "number of columns != 2 at line: " << line;
    
    string fn = tuple[0];
    labels = tuple[1];
    image_db_label = is_multilabel ? 0 : category_mapping[labels];
    
    // Process image
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(fn,
        image_db_label, resize_height, resize_width, is_color,
        enc, &image_datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = image_datum.channels() * image_datum.height() * image_datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = image_datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }

    // Process multilabel
    std::vector<std::string> multilabels;
    Datum label_datum;

    if (is_multilabel) {
      boost::split(multilabels, labels, boost::is_any_of(","));
      label_datum.set_channels(category_count);
      label_datum.set_height(1);
      label_datum.set_width(1);
      
      for (int i = 0; i < category_count; i++)
        label_datum.add_float_data(0);
      
      for (int i = 0; i < multilabels.size(); i++) {
        CHECK(category_mapping.find(multilabels[i]) != category_mapping.end()) << "Unknown category at line: " << line;
        label_datum.set_float_data(category_mapping[multilabels[i]], 1);
      }
    }
    
    // Transaction
    string out;
    CHECK(image_datum.SerializeToString(&out));
    image_db_txn->Put(caffe::format_int(count, 8), out);
    if (is_multilabel) {
      CHECK(label_datum.SerializeToString(&out));
      labels_db_txn->Put(caffe::format_int(count, 8), out);
    }

    if (++count % 1000 == 0) {
      image_db_txn->Commit();
      image_db_txn.reset(image_db->NewTransaction());

      if (is_multilabel) {
        labels_db_txn->Commit();
        labels_db_txn.reset(labels_db->NewTransaction());
      }
      LOG(ERROR) << "Processed " << count << " files correctly.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    image_db_txn->Commit();
    if (is_multilabel)
      labels_db_txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}

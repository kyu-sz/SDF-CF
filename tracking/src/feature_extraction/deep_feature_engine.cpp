#include <caffe2/core/predictor.h>

#include "deep_feature_engine.h"

using namespace caffe2;
using namespace std;

bool deep_engine_init(void **deep_engine_handle) {
    // TODO
    return false;
}

bool deep_engine_free(void *deep_engine_handle) {
    delete static_cast<Predictor *>(deep_engine_handle);
    return false;
}
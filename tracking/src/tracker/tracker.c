#include "tracker.h"
#include <feature_extraction/deep_feature_engine.h>
#include <stdlib.h>

typedef struct {
  void *deep_engine_handle;
} TrackerHandle;

bool tracker_init(void **tracker_handle, const ImageBGR *frame, const Rect *bndbox) {
    TrackerHandle *handle = malloc(sizeof(TrackerHandle));
    deep_engine_init(&handle->deep_engine_handle);
    *tracker_handle = handle;
    // TODO
    return false;
}

bool tracker_track(void *tracker_handle, const ImageBGR *frame, Rect *bndbox) {
    // TODO
    return false;
}

bool tracker_reset(void *tracker_handle, const ImageBGR *frame, const Rect *bndbox) {
    // TODO
    return false;
}

bool tracker_free(void *tracker_handle) {
    TrackerHandle *handle = tracker_handle;
    deep_engine_free(handle->deep_engine_handle);
    // TODO
    return false;
}
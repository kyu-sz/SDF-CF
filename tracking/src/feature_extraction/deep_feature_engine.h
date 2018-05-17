#include <stdbool.h>
#include <tracker/tracker.h>

#ifndef DEEP_FEATURE_EXTRACTOR_H
#define DEEP_FEATURE_EXTRACTOR_H

#pragma once

bool deep_engine_init(void **deep_engine_handle);

bool deep_engine_free(void *deep_engine_handle);

bool deep_engine_forward(void *deep_engine_handle, const Rect *img);

bool deep_engine_get_layer(void *deep_engine_handle, const char *layers[], float *features[]);

#endif //DEEP_FEATURE_EXTRACTOR_H
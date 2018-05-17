//
// Created by kaiy1 on 5/16/18.
//

#include <stdbool.h>

#ifndef TRACKER_H
#define TRACKER_H

#pragma once

/// @brief A rectangle area.
/// Structure for specifying a rectangle area in an image.
typedef struct {
  int x;    /// X coordinate of the left-upper corner.
  int y;    /// Y coordinate of the left-upper corner.
  int w;    /// Width of the rectangle.
  int h;    /// Height of the rectangle.
} Rect;

/// @brief A BGR image.
/// Structure of an image in BGR pixel order.
typedef struct {
  int w;    /// Width of the image.
  int h;    /// Height of the image.
  const unsigned char *data;    /// Pixel data of the image, in the same format as OpenCV.
} ImageBGR;

/// @brief Initialize a tracker.
/// Initialize a tracker with an initial frame and a bounding box of a target in the frame,
/// and return the tracker handle.
/// @param tracker_handle pointer of pointer for storing the tracker handle.
/// @param frame image of the initial frame.
/// @param bndbox bounding box of a target in the initial frame.
/// @return Boolean indicating the initialization succeeds or not.
bool tracker_init(void **tracker_handle, const ImageBGR *frame, const Rect *bndbox);

/// @brief Track a frame.
/// Predict the bounding box of the target in the frame.
/// @param tracker_handle tracker handle.
/// @param frame image of the frame.
/// @param bndbox pointer to the buffer of the bounding box of the target.
/// @return Boolean indicating the tracking succeeds or not.
bool tracker_track(void *tracker_handle, const ImageBGR *frame, Rect *bndbox);

/// @brief Reset the tracker.
/// Reset a tracker with a frame and a bounding box of a target in the frame.
/// @param tracker_handle tracker handle.
/// @param frame image of the frame.
/// @param bndbox bounding box of a target in the frame.
/// @return Boolean indicating the resetting succeeds or not.
bool tracker_reset(void *tracker_handle, const ImageBGR *frame, const Rect *bndbox);

/// @brief Free the tracker.
/// Free the tracker.
/// @param tracker_handle tracker handle.
/// @return Boolean indicating the freeing operation succeeds or not.
bool tracker_free(void *tracker_handle);

#endif //TRACKER_H

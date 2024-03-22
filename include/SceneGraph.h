#ifndef SCENEGRAPH_H
#define SCENEGRAPH_H

#include "ImageLoader.h"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

struct ImageNode {
  int id;            // Image ID or index
  ImageWithID image; // The image itself with the camera ID
};

struct ImageEdge {
  int fromId;                      // Source image ID
  int toId;                        // Destination image ID
  std::vector<cv::DMatch> matches; // Inlier matches between the two images
  cv::Mat H; // Homography matrix representing the geometric relation
};

struct SceneGraph {
  std::vector<ImageNode> nodes;
  std::vector<ImageEdge> edges;
};

#endif

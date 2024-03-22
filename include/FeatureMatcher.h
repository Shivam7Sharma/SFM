// This is a placeholder for your feature matching module.
// include/FeatureMatcher.h
#pragma once
#include "ImageLoader.h"
#include "SceneGraph.h"
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

struct FeatureMatchResult {
  SceneGraph graph;
  std::vector<std::vector<cv::KeyPoint>> allKeypoints;
  std::vector<cv::Mat> allDescriptors;
};

// Placeholder for feature matching function
FeatureMatchResult matchFeatures(const std::vector<ImageWithID> &images);

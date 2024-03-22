// SfMInitializer.h
#ifndef SFMINITIALIZER_H
#define SFMINITIALIZER_H

#include "CameraParameters.h" // Ensure this is correctly included
#include "SceneGraph.h"
#include <opencv2/core.hpp>
#include <vector>

class SfMInitializer {
public:
  SfMInitializer();
  void initializeReconstruction(
      SceneGraph &sceneGraph, const std::vector<CameraParameters> &cameraParams,
      const std::vector<std::vector<cv::KeyPoint>> &allKeypoints,
      const std::vector<cv::Mat> &allDescriptors);
};

#endif // SFMINITIALIZER_H

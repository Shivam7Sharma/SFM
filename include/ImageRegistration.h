// ImageRegistration.h
#ifndef IMAGEREGISTRATION_H
#define IMAGEREGISTRATION_H

#include "CameraParameters.h" // Correct include statement
#include "SceneGraph.h"
#include <opencv2/core.hpp>
#include <vector>

class ImageRegistration {
public:
  void registerImages(SceneGraph &sceneGraph,
                      const std::vector<cv::Mat> &images,
                      const std::vector<CameraParameters> &cameraParams);

private:
  cv::Mat registerImage(const cv::Mat &image,
                        const std::vector<cv::KeyPoint> &keypoints,
                        const cv::Mat &descriptors,
                        const CameraParameters &cameraParam);
};

#endif // IMAGEREGISTRATION_H

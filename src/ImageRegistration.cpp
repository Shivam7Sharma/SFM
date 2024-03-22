// ImageRegistration.cpp
#include "ImageRegistration.h"
#include <opencv2/opencv.hpp>
#include <opencv2/sfm.hpp>

void ImageRegistration::registerImages(
    SceneGraph &sceneGraph, const std::vector<cv::Mat> &images,
    const std::vector<CameraParameters> &cameraParams) {
  // Ensure there's a camera parameter set for each image
  if (images.size() != cameraParams.size()) {
    std::cerr << "Error: The number of images does not match the number of "
                 "camera parameter sets."
              << std::endl;
    return;
  }

  for (size_t i = 0; i < images.size(); ++i) {
    // Placeholder for feature detection
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Here, you would detect keypoints and compute descriptors for images[i]
    // For demonstration, this step is assumed

    // Register the image using its corresponding camera parameters
    cv::Mat pose =
        registerImage(images[i], keypoints, descriptors, cameraParams[i]);

    // Update sceneGraph with the pose of the newly registered image
    // This step is left as an exercise; you'd likely want to add the pose to
    // the SceneGraph
  }
}

cv::Mat ImageRegistration::registerImage(
    const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints,
    const cv::Mat &descriptors, const CameraParameters &cameraParam) {
  // Convert CameraParameters to the format expected by solvePnP, if necessary
  cv::Mat cameraMatrix =
      (cv::Mat_<double>(3, 3) << cameraParam.fx, 0, cameraParam.cx, 0,
       cameraParam.fy, cameraParam.cy, 0, 0, 1);
  cv::Mat distCoeffs = cv::Mat::zeros(
      4, 1, CV_64F); // Assuming no distortion; adjust if available

  // Convert keypoints to Point2f for solvePnP
  std::vector<cv::Point2f> points2f;
  cv::KeyPoint::convert(keypoints, points2f);

  // Placeholder for 3D points corresponding to keypoints; to be filled based on
  // your scene reconstruction
  std::vector<cv::Point3f> objectPoints;

  // Placeholder: Load objectPoints corresponding to keypoints from your scene
  // reconstruction

  if (points2f.size() < 4 || objectPoints.size() < 4) {
    std::cerr << "Not enough points for PnP." << std::endl;
    return cv::Mat();
  }

  cv::Mat rvec, tvec;
  // SolvePnP to find the pose of the camera
  bool success = cv::solvePnP(objectPoints, points2f, cameraMatrix, distCoeffs,
                              rvec, tvec);

  if (!success) {
    std::cerr << "PnP solution not found." << std::endl;
    return cv::Mat();
  }

  // Construct the pose matrix (3x4 RT matrix) from rvec and tvec
  cv::Mat pose;
  cv::Rodrigues(rvec, pose); // Convert rotation vector to matrix
  pose.resize(3, 4);         // Resize to 3x4 to append translation vector
  tvec.copyTo(pose.col(3));  // Append translation vector

  return pose;
}

// SfMInitializer.cpp
#include "SfMInitializer.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp> // Include this at the top with other includes

SfMInitializer::SfMInitializer() {}

void SfMInitializer::initializeReconstruction(
    SceneGraph &sceneGraph, const std::vector<CameraParameters> &cameraParams,
    const std::vector<std::vector<cv::KeyPoint>> &allKeypoints,
    const std::vector<cv::Mat> &allDescriptors) {

  if (sceneGraph.edges.empty()) {
    std::cerr << "SceneGraph does not contain any edges/matches." << std::endl;
    return;
  }

  // Function to find camera parameters by camera_id
  auto findCameraParams = [&](int image_id) -> const CameraParameters * {
    for (const auto &params : cameraParams) {
      if (params.camera_id == sceneGraph.nodes[image_id].image.camera_id) {
        return &params;
      }
    }
    return nullptr; // Return nullptr if not found
  };

  int bestPair[2] = {-1, -1};
  size_t maxMatches = 0;

  int bestEdgeIndex = -1; // Add this variable to store the index of the edge
                          // with the highest number of matches

  // Iterate through edges to find the pair with the highest number of matches
  for (size_t i = 0; i < sceneGraph.edges.size(); ++i) {
    if (sceneGraph.edges[i].matches.size() > maxMatches) {
      maxMatches = sceneGraph.edges[i].matches.size();
      bestPair[0] = sceneGraph.edges[i].fromId;
      bestPair[1] = sceneGraph.edges[i].toId;
      bestEdgeIndex = i; // Save the index of the edge
    }
  }

  if (bestEdgeIndex != -1) {
    // Retrieve images of the best pair
    cv::Mat img1 = sceneGraph.nodes[bestPair[0]].image.image;
    cv::Mat img2 = sceneGraph.nodes[bestPair[1]].image.image;

    // Retrieve keypoints of the best pair
    const std::vector<cv::KeyPoint> &keypoints1 = allKeypoints[bestPair[0]];
    const std::vector<cv::KeyPoint> &keypoints2 = allKeypoints[bestPair[1]];

    // Use bestEdgeIndex to retrieve the matches directly
    std::vector<cv::DMatch> bestMatches =
        sceneGraph.edges[bestEdgeIndex].matches;

    // Draw matches
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, bestMatches, imgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Display the matches
    cv::namedWindow("Matches between best pair", cv::WINDOW_NORMAL);
    cv::imshow("Matches between best pair", imgMatches);
    cv::waitKey(0); // Wait for a key press to close the window
  } else {
    std::cerr << "Failed to find a suitable pair for visualization."
              << std::endl;
    return;
  }

  // Fetch the camera parameters for the best pair using camera_id
  const CameraParameters *params1 = findCameraParams(bestPair[0]);
  const CameraParameters *params2 = findCameraParams(bestPair[1]);

  if (!params1 || !params2) {
    std::cerr << "Camera parameters for the best pair not found." << std::endl;
    return;
  }

  // Now you can construct the camera matrix for each camera using the found
  // parameters
  cv::Mat K1 = (cv::Mat_<double>(3, 3) << params1->fx, 0, params1->cx, 0,
                params1->fy, params1->cy, 0, 0, 1);
  std::cout << "K1: " << K1 << std::endl;
  cv::Mat K2 = (cv::Mat_<double>(3, 3) << params2->fx, 0, params2->cx, 0,
                params2->fy, params2->cy, 0, 0, 1);
  std::cout << "K2: " << K2 << std::endl;

  // Extract matching points using the keypoints from the best pair
  std::vector<cv::Point2f> points1, points2;
  for (auto &match : sceneGraph.edges[bestEdgeIndex].matches) {
    points1.push_back(allKeypoints[bestPair[0]][match.queryIdx].pt);
    points2.push_back(allKeypoints[bestPair[1]][match.trainIdx].pt);
  }
  std::cout << "Extracted " << points1.size() << " matching points."
            << std::endl;

  // Compute the essential matrix from the points using the first camera matrix
  // (assuming similar camera settings for simplicity)
  cv::Mat E = cv::findEssentialMat(points1, points2, K1, cv::RANSAC, 0.999, 1.0,
                                   cv::noArray());

  std::cout << "Essential matrix computed." << std::endl;

  // Recover the pose from the essential matrix
  cv::Mat R, t;
  cv::recoverPose(E, points1, points2, K1, R, t, cv::noArray());
  std::cout << "Pose recovered. R: " << R << " t: " << t << std::endl;

  // Triangulate initial 3D points
  cv::Mat points4D;
  // Correct definition of P1 to include K1
  cv::Mat P1 = K1 * cv::Mat::eye(3, 4, CV_64F); // This incorporates K1 into P1
  cv::Mat P2(3, 4, CV_64F);                     // Camera 2's projection matrix
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.col(3));
  P2 = K2 * P2; // Adjusting based on Camera 2's parameters

  cv::triangulatePoints(P1, P2, points1, points2, points4D);
  std::cout << "Points triangulated." << std::endl;
  std::cout << "points4D size: " << points4D.size() << std::endl;
  std::cout << "points4D type: " << points4D.type() << std::endl;

  // Convert 4D homogeneous points to 3D
  cv::Mat points3D;
  cv::convertPointsFromHomogeneous(points4D.t(), points3D);
  std::cout << "Triangulated " << points3D.total() << " 3D points."
            << std::endl;

  // points3D now contains the triangulated 3D points
  // R and t contain the relative rotation and translation from camera 1 to
  // camera 2// At the end of initializeReconstruction function, add the
  // visualization code

  // Check if OpenCV Viz module is available and points3D is not empty
  // Check if OpenCV Viz module is available and points3D is not empty
  // Initialize the Viz window
  cv::viz::Viz3d window("Triangulated Points Visualization");

  // Check if points3D is not empty and the Viz window has not been stopped
  if (!points3D.empty() && !window.wasStopped()) {
    window.setBackgroundColor(cv::viz::Color::black());

    // Convert points3D to a format suitable for visualization
    std::vector<cv::Vec3f> pointCloud;
    for (int i = 0; i < points3D.rows; i++) {
      // Extract each point and push it into the pointCloud vector
      cv::Point3f pt = points3D.at<cv::Vec3f>(i, 0);
      pointCloud.push_back(pt);
    }

    // Create a WCloud widget for point cloud visualization
    cv::viz::WCloud cloudWidget(pointCloud, cv::viz::Color::green());
    window.showWidget("Point Cloud", cloudWidget);

    // Add coordinate axes to help visualize the orientation of the point cloud
    window.showWidget("Coordinate Axes", cv::viz::WCoordinateSystem());

    // Display the visualization and wait for the window to be closed
    window.spin();
  }
}

#include "FeatureMatcher.h"
#include "ImageLoader.h"
#include "SceneGraph.h" // Include this if you're using a separate header file
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

FeatureMatchResult matchFeatures(const std::vector<ImageWithID> &images) {
  FeatureMatchResult result;
  if (images.size() < 2) {
    std::cerr << "Need at least two images to match features.\n";
    return result; // Return an empty result object
  }

  // Use ORB here as SIFT is not directly available in CUDA. Adjust as necessary
  // for your application.
  cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create();
  auto matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

  for (int i = 0; i < images.size(); ++i) {
    cv::cuda::GpuMat gpuImage(images[i].image);

    cv::cuda::GpuMat gpuKeypoints, gpuDescriptors;
    // Asynchronously detect and compute keypoints and descriptors on the GPU
    detector->detectAndComputeAsync(gpuImage, cv::noArray(), gpuKeypoints,
                                    gpuDescriptors);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->convert(gpuKeypoints, keypoints);
    gpuDescriptors.download(descriptors);

    result.graph.nodes.push_back({i, images[i]});
    result.allKeypoints.push_back(keypoints);
    result.allDescriptors.push_back(descriptors);
  }

  // Match features between each pair of images
  for (size_t i = 0; i < images.size(); ++i) {
    for (size_t j = i + 1; j < images.size(); ++j) {
      cv::cuda::GpuMat gpuDescriptorsI(result.allDescriptors[i]);
      cv::cuda::GpuMat gpuDescriptorsJ(result.allDescriptors[j]);
      std::vector<std::vector<cv::DMatch>> knnMatches;
      matcher->knnMatch(gpuDescriptorsI, gpuDescriptorsJ, knnMatches,
                        2); // Find the 2 nearest matches for Lowe's ratio test

      // Filter matches using Lowe's ratio test
      const float ratioThresh = 0.6f;
      std::vector<cv::DMatch> goodMatches;
      for (const auto &match : knnMatches) {
        if (match.size() == 2 &&
            match[0].distance < ratioThresh * match[1].distance) {
          goodMatches.push_back(match[0]);
        }
      }

      // Similar processing for inliers as before
      // Note: Homography check is a CPU operation, so it is not accelerated by
      // CUDA
      if (!goodMatches.empty()) {
        std::vector<cv::Point2f> srcPoints, dstPoints;
        for (const auto &match : goodMatches) {
          srcPoints.push_back(result.allKeypoints[i][match.queryIdx].pt);
          dstPoints.push_back(result.allKeypoints[j][match.trainIdx].pt);
        }
        std::vector<cv::DMatch> inlierMatches;
        if (srcPoints.size() >= 4 && dstPoints.size() >= 4) {
          std::vector<char> inliersMask;
          cv::Mat H = cv::findHomography(srcPoints, dstPoints, cv::RANSAC, 3,
                                         inliersMask);

          for (size_t k = 0; k < inliersMask.size(); ++k) {
            if (inliersMask[k]) {
              inlierMatches.push_back(goodMatches[k]);
            }
          }
          // Update the SceneGraph with the matches considered inliers
          ImageEdge edge;
          edge.fromId = i;
          edge.toId = j;
          edge.matches = inlierMatches; // Storing only inlier matches
          edge.H = H; // Homography matrix calculated from inliers
          result.graph.edges.push_back(edge);
        } else {
          std::cerr << "Not enough matches for findHomography" << std::endl;
        }
      }
    }
  }

  return result;
}

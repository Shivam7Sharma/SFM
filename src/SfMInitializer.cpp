// SfMInitializer.cpp
// SfMInitializer.cpp
#include "SfMInitializer.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <iostream>
#include <memory>
#include <memory> // for std::make_unique
#include <opencv2/core/eigen.hpp>
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

  // Print all the 3D points
  for (int i = 0; i < points3D.rows; i++) {
    cv::Point3f point = points3D.at<cv::Point3f>(i, 0);
    std::cout << "Point " << i << ": (" << point.x << ", " << point.y << ", "
              << point.z << ")" << std::endl;
  }

  g2o::SparseOptimizer optimizer;
  std::cout << "SparseOptimizer created.\n";
  optimizer.setVerbose(true);
  // Create the linear solver
  auto linearSolver = std::make_unique<g2o::LinearSolverDense<
      g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>::PoseMatrixType>>();
  std::cout << "LinearSolver created.\n";

  // Create the block solver
  auto blockSolver =
      std::make_unique<g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>>(
          std::move(linearSolver));
  std::cout << "BlockSolver created.\n";

  // Set the optimization algorithm
  optimizer.setAlgorithm(
      new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));
  std::cout << "Optimization algorithm set.\n";

  // Create a new g2o::CameraParameters object
  g2o::CameraParameters *cam_params = new g2o::CameraParameters(
      params1->fx, Eigen::Vector2d(params1->cx, params1->cy), 0.);

  // Set the id from your CameraParameters struct
  cam_params->setId(params1->camera_id);
  std::cout << "params1 fx, cx, cy, and id : " << params1->fx << " "
            << params1->cx << " " << params1->cy << " " << params1->camera_id
            << std::endl;

  std::cout << "Camera focal length " << cam_params->focal_length
            << "\n Camera Principle Point " << cam_params->principle_point
            << std::endl;
  // Add the camera parameters to the optimizer
  if (!optimizer.addParameter(cam_params)) {
    assert(false);
  }
  if (params2->camera_id != params1->camera_id) {
    std::cerr << "Camera IDs are different.\n";

    // Create a new g2o::CameraParameters object for the second camera
    g2o::CameraParameters *cam_params2 = new g2o::CameraParameters(
        params2->fx, Eigen::Vector2d(params2->cx, params2->cy), 0.);

    std::cout << "params2 fx, cx, cy, and id : " << params2->fx << " "
              << params2->cx << " " << params2->cy << " " << params2->camera_id
              << std::endl;

    // Set the id from your CameraParameters struct
    cam_params2->setId(params2->camera_id);

    // Add the camera parameters to the optimizer
    if (!optimizer.addParameter(cam_params2)) {
      assert(false);
    }
  } else {
    std::cout << "Camera IDs are the same.\n";
  }

  // Convert from cv::Mat to Eigen::Matrix, check for validity of conversion
  Eigen::Matrix3d R_eigen;
  cv::cv2eigen(R, R_eigen);
  Eigen::Vector3d t_eigen;
  cv::cv2eigen(t, t_eigen);
  std::cout << "Converted R and t from cv::Mat to Eigen.\n";
  g2o::SE3Quat pose2 = g2o::SE3Quat(R_eigen, t_eigen);

  g2o::SE3Quat pose1 = (pose2).inverse();
  std::cout << "Inverse pose computed.\n";

  auto *vertex0 = new g2o::VertexSE3Expmap();
  vertex0->setId(0);           // Set the ID to 0 for the first camera pose
  vertex0->setEstimate(pose1); // Set the initial estimate for the pose
  optimizer.addVertex(vertex0);
  std::cout << "Added first camera pose as vertex.\n";
  auto *vertex1 = new g2o::VertexSE3Expmap();
  vertex1->setId(1);           // Set the ID to 0 for the first camera pose
  vertex1->setEstimate(pose2); // Set the initial estimate for the pose
  optimizer.addVertex(vertex1);
  std::cout << "Added second camera pose as vertex.\n";
  // Add nodes and edges to the optimizer here

  // points3D now contains the triangulated 3D points
  // R and t contain the relative rotation and translation from camera 1 to
  // camera 2// At the end of initializeReconstruction function, add the
  // visualization code
  // Assuming points is a std::vector<cv::Point3f> containing your 3D points

  // Check for segmentation fault around adding vertices for 3D points
  int edgeId = 0;
  for (int i = 0; i < points3D.rows; ++i) {
    auto *pointVertex = new g2o::VertexPointXYZ();
    pointVertex->setId(i + 2);
    // Ensure the point data is valid before setting it
    Eigen::Vector3d point =
        Eigen::Vector3d(points3D.at<double>(i, 0), points3D.at<double>(i, 1),
                        points3D.at<double>(i, 2));
    if (!std::isnan(point[0]) && !std::isnan(point[1]) &&
        !std::isnan(point[2])) {
      pointVertex->setEstimate(point);
      pointVertex->setMarginalized(true);
      optimizer.addVertex(pointVertex);
      // std::cout << "Added 3D point as vertex: " << i + 1 << "\n";
    } else {
      // std::cout << "Invalid 3D point encountered at index: " << i << "\n";
    }
    // Similar checks for edges
    for (auto pose : {0, 1}) {
      // const auto &match = sceneGraph.edges[bestEdgeIndex].matches[i];
      auto *edge = new g2o::EdgeProjectXYZ2UV();
      // std::cout << "Attempting to create edge for match index: " << i
      //           << std::endl;

      // Ensure the keypoints indices are within bounds
      // if (match.queryIdx >= allKeypoints[bestPair[0]].size() ||
      // match.trainIdx >= allKeypoints[bestPair[1]].size()) {
      // std::cerr << "KeyPoint index out of bounds for match index: " << i
      // << std::endl;
      // continue; // Skip this match
      // }
      // std::cout << "Match Query Index: " << match.queryIdx << std::endl;
      // // Ensure vertices exist for the given indices
      g2o::OptimizableGraph::Vertex *vertex0 =
          dynamic_cast<g2o::OptimizableGraph::Vertex *>(
              optimizer.vertex(i + 2));
      // std::cout << "passed vertex0\n";
      g2o::OptimizableGraph::Vertex *vertex1 =
          dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pose));
      // std::cout << "passed pose vertex\n";
      if (!vertex0 || !vertex1) {
        std::cerr << "Vertex missing for match index: " << i << std::endl;
        continue; // Skip this match
      }

      if (vertex0 == nullptr) {
        std::cout << "Vertex 0 is nullptr.\n";
      } else if (vertex1 == nullptr) {
        std::cout << "Vertex 1 is nullptr.\n";
      }
      // std::cout << "Vertices found for match index: " << i + 2 << std::endl;
      // Set vertices for the edge
      edge->setVertex(0, vertex0);
      // std::cout << "Set vertex 0\n";
      edge->setVertex(1, vertex1);
      // std::cout << "Set vertex 1\n";
      edge->setId(edgeId); // Set the ID of the edge

      // Set measurement and information
      if (pose == 0) {
        edge->setMeasurement(
            Eigen::Vector2d(points1[i].x, points1[i].y)); // Use the first image
        edge->setParameterId(0, params1->camera_id);
      } else {
        edge->setMeasurement(Eigen::Vector2d(
            points2[i].x, points2[i].y)); // Use the second image
        edge->setParameterId(0, params2->camera_id);
      }
      // std::cout << "Set measurement\n";
      edge->setInformation(Eigen::Matrix2d::Identity());
      // std::cout << "Set information\n";

      if (edge == nullptr) {
        std::cout << "Edge is nullptr.\n";
      } else if (optimizer.vertex(edge->vertices()[0]->id()) == nullptr ||
                 optimizer.vertex(edge->vertices()[1]->id()) == nullptr) {
        std::cout << "One or both vertices do not exist in the optimizer.\n";
      }

      if (optimizer.solver() == nullptr) {
        std::cout << "No solver set for the optimizer.\n";
      } else {
        // std::cout << "Solver set for the optimizer.\n";
      }

      // std::cout << "Number of vertices in the optimizer: "
      //           << optimizer.vertices().size() << "\n";
      // std::cout << "Number of edges in the optimizer: "
      //           << optimizer.edges().size() << "\n";

      if (edge->vertices()[0] != nullptr && edge->vertices()[1] != nullptr) {
        // Check if they exist within the optimizer
        if (optimizer.vertex(edge->vertices()[0]->id()) != nullptr &&
            optimizer.vertex(edge->vertices()[1]->id()) != nullptr) {
          std::cout << "Both vertices exist within the optimizer.\n";
        } else {
          std::cout
              << "One or both vertices do not exist within the optimizer.\n";
        }
      } else {
        std::cout << "One or both vertices are null.\n";
      }

      // std::cout << optimizer << std::endl;
      // Attempt to add edge to optimizer
      if (!optimizer.addEdge(edge)) {
        std::cerr << "Failed to add edge to optimizer for match index: " << i
                  << std::endl;
        delete edge; // Cleanup if adding edge failed
      } else {
        // std::cout << "Successfully added edge for match index: " << i
        //           << std::endl;
      }
      // std::cout << "Finished processing match index: " << i << std::endl;
    }
  }
  // std::cout << "Added all 3D points as vertices.\n";

  optimizer.initializeOptimization();
  optimizer.optimize(10); // The number of iterations can be adjusted as needed
  std::cout << "Optimization complete.\n";
  // Create a new cv::Mat to store the optimized 3D points
  cv::Mat optimizedPoints3D(points3D.rows, points3D.cols, points3D.type());
  std::cout << "Created optimizedPoints3D.\n";
  // Retrieve the optimized 3D points from the optimizer
  for (int i = 0; i < points3D.rows; ++i) {
    g2o::VertexPointXYZ *v =
        static_cast<g2o::VertexPointXYZ *>(optimizer.vertex(i + 2));
    Eigen::Vector3d point = v->estimate();
    optimizedPoints3D.at<cv::Vec3d>(i, 0) =
        cv::Vec3d(point[0], point[1], point[2]);
  }
  std::cout << "Retrieved optimized 3D points.\n";

  // Retrieve the optimized camera pose from the optimizer
  g2o::VertexSE3Expmap *v =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat pose = v->estimate();
  std::cout << "Retrieved optimized camera pose.\n";
  // Convert the pose to rotation matrix and translation vector

  cv::Mat R_cv, t_cv;
  cv::eigen2cv(pose.rotation().toRotationMatrix(), R_cv);
  cv::eigen2cv(pose.translation(), t_cv);
  std::cout << "Converted optimized pose to cv::Mat.\n";
  // Check if OpenCV Viz module is available and points3D is not empty
  // Check if OpenCV Viz module is available and points3D is not empty
  // Initialize the Viz window
  cv::viz::Viz3d window("Triangulated Points Visualization");

  // Check if points3D is not empty and the Viz window has not been stopped
  if (!optimizedPoints3D.empty() && !window.wasStopped()) {
    window.setBackgroundColor(cv::viz::Color::black());

    // Convert points3D to a format suitable for visualization
    std::vector<cv::Vec3f> pointCloud;
    for (int i = 0; i < optimizedPoints3D.rows; i++) {
      // Extract each point and push it into the pointCloud vector
      std::cout << "Optimized 3D point: "
                << optimizedPoints3D.at<cv::Vec3d>(i, 0) << std::endl;
      cv::Point3f pt = optimizedPoints3D.at<cv::Vec3f>(i, 0);
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
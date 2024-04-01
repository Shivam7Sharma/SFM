// include/CameraParameters.h
#pragma once
#include <string>
#include <vector>

struct CameraParameters {
  int camera_id;
  std::string model;
  int width, height;
  double fx, fy, cx, cy;
};

std::vector<CameraParameters> loadCameraParameters(const std::string &filename);

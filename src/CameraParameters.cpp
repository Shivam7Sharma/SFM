// src/CameraParameters.cpp
#include "CameraParameters.h"
#include <fstream>
#include <iostream>
#include <sstream>

std::vector<CameraParameters>
loadCameraParameters(const std::string &filename) {
  std::vector<CameraParameters> cameraParams;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error opening camera parameters file." << std::endl;
    return cameraParams;
  }

  while (getline(file, line)) {
    if (line[0] == '#' || line.empty())
      continue;
    std::istringstream iss(line);
    CameraParameters params;
    iss >> params.camera_id >> params.model >> params.width >> params.height >>
        params.fx >> params.fy >> params.cx >> params.cy;
    cameraParams.push_back(params);
  }
  file.close();
  return cameraParams;
}

// include/ImageLoader.h
#pragma once
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

struct ImageWithID {
  cv::Mat image;
  int camera_id;
};

std::vector<ImageWithID> loadImagesFromFolder(const std::string &folderPath);

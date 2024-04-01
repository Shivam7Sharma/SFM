#include "ImageLoader.h"
#include <algorithm>
#include <cctype>
#include <filesystem> // Require C++17
#include <opencv2/opencv.hpp>
#include <vector>

namespace fs = std::filesystem;

std::vector<ImageWithID> loadImagesFromFolder(const std::string &folderPath) {
  std::vector<ImageWithID> images;

  for (const auto &entry : fs::recursive_directory_iterator(folderPath)) {
    if (entry.is_regular_file()) {
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
          // Assuming folder names are integers representing camera IDs
          int camera_id =
              std::stoi(entry.path().parent_path().filename().string());
          images.push_back({image, camera_id});
        }
      }
    }
  }
  return images;
}

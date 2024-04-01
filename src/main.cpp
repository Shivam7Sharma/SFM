#include "CameraParameters.h"
#include "FeatureMatcher.h"
#include "ImageLoader.h"
#include "ImageRegistration.h"
#include "SfMInitializer.h"
#include <iostream>

int main() {
  // Example usage
  std::string paramsFile = "/home/shivam/Computer_Vision/SFM/3d_reconstruction/"
                           "delivery_area_rig_undistorted/delivery_area/"
                           "rig_calibration_undistorted/cameras.txt";
  auto cameraParams = loadCameraParameters(paramsFile);

  std::string folderPath = "/home/shivam/Computer_Vision/SFM/3d_reconstruction/"
                           "delivery_area_rig_undistorted/delivery_area/images";
  std::vector<ImageWithID> images = loadImagesFromFolder(folderPath);

  // Generate SceneGraph with feature matches
  FeatureMatchResult matchResult = matchFeatures(images);

  // Initialize the reconstruction process
  // Assuming you've called matchFeatures and have allKeypoints and
  // allDescriptors available
  SfMInitializer initializer;
  initializer.initializeReconstruction(matchResult.graph, cameraParams,
                                       matchResult.allKeypoints,
                                       matchResult.allDescriptors);

  // ImageRegistration imgReg;
  // imgReg.registerImages(matchResult.graph, images, cameraParams);

  return 0;
}

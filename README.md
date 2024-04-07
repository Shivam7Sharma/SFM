# Structure from Motion (SfM) (In development)

This project contains the implementation of a Structure from Motion (SfM) initializer. The SfM initializer takes a set of images, detects features in these images, matches these features across images, and then uses these matches to estimate the 3D structure of the scene and the camera poses.

## File Structure

The main file in this project is `SfMInitializer.cpp`. This file contains the implementation of the SfM initializer.

## How it Works

The SfM initializer works by first detecting features in each image using a feature detector. These features are then matched across images using a feature matcher. The matches are then used to estimate the 3D structure of the scene and the camera poses.

After the 3D points and camera poses have been estimated, the SfM initializer optimizes these estimates using a bundle adjustment algorithm. The bundle adjustment algorithm minimizes the reprojection error, which is the difference between the observed positions of features in the images and the positions predicted by the camera model and the estimated 3D points and camera poses.

Finally, the optimized 3D points are visualized using the `cv::viz` module in OpenCV. The 3D points are displayed in a 3D window, along with coordinate axes to help visualize the orientation of the point cloud.

## Usage

To use the SfM initializer, you need to provide a set of images of a scene from different viewpoints. The images should be in a format that can be read by OpenCV, such as JPEG or PNG.

## Dependencies

This project depends on the following libraries:

- CUDA
- Eigen
- OpenCV: Used for image processing and visualization.
- g2o: Used for the bundle adjustment optimization.

## Building and Running

To build and run the project, follow these steps:

1. Clone the repository.
2. Navigate to the project directory.
3. Build the project using CMake.
4. Run the executable.

## License

This project is licensed under the MIT License.
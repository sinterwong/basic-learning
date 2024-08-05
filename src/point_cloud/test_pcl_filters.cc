#include <iostream>
#include <pcl/common/random.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main() {
  // Create a point cloud with random data
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width = 1000;
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width * cloud->height);

  pcl::common::UniformGenerator<float> gen(-1.0f, 1.0f);
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    cloud->points[i].x = gen.run();
    cloud->points[i].y = gen.run();
    cloud->points[i].z = gen.run();
  }

  std::cout << "Generated " << cloud->width * cloud->height
            << " data points with the following fields: " << std::endl;

  // Create a VoxelGrid filter object
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(cloud);
  voxel_grid.setLeafSize(0.01f, 0.01f, 0.01f); // Set the voxel grid size (1cm)

  // Apply the filter to downsample the point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
      new pcl::PointCloud<pcl::PointXYZ>);
  voxel_grid.filter(*cloud_filtered);

  // Print the number of points in the original and filtered point clouds
  std::cout << "PointCloud before filtering: " << cloud->width * cloud->height
            << " data points (" << pcl::getFieldsList(*cloud) << ")."
            << std::endl;
  std::cout << "PointCloud after filtering: "
            << cloud_filtered->width * cloud_filtered->height
            << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")."
            << std::endl;

  // Save the filtered point cloud to a new PCD file
  pcl::io::savePCDFileASCII("output.pcd", *cloud_filtered);
  std::cerr << "Saved " << cloud_filtered->points.size()
            << " data points to output.pcd." << std::endl;

  return 0;
}


#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:
	// Elements
	// ********

	int count;
	PointXYZ centroid; // 所有点位置累加，计算质心所需

	vector<PointXYZ> points;
	vector<vector<float>> points_features;
	vector<vector<int>> points_labels;

	// Methods
	// *******
	// Constructor
	SampledData() 
	{ 
		count = 0; 
		centroid = PointXYZ();
	}

	SampledData(const size_t fdim, const size_t ldim)
	{
		count = 0;
		centroid = PointXYZ();
	    points_features.reserve(fdim);
		points_labels.reserve(ldim);
	}

	// Method Update
	void update_all(const PointXYZ p, vector<float>::iterator f_begin, vector<int>::iterator l_begin, size_t fdim, size_t ldim)
	{
		count += 1;
		centroid += p;

		points.push_back(p);
		// Store the current point's features and labels
		points_features.emplace_back(f_begin, f_begin + fdim);
		points_labels.emplace_back(l_begin, l_begin + ldim);
		return;
	}

	void update_features(const PointXYZ p, vector<float>::iterator f_begin, size_t fdim, size_t ldim)
	{
		count += 1;
		centroid += p;

		points.push_back(p);
		// Store the current point's features and labels
		points_features.emplace_back(f_begin, f_begin + fdim);
		return;
	}

	void update_classes(const PointXYZ p, vector<int>::iterator l_begin, size_t fdim, size_t ldim)
	{
		count += 1;
		centroid += p;

		points.push_back(p);
		// Store the current point's features and labels
		points_labels.emplace_back(l_begin, l_begin + ldim);
		return;
	}

	void update_points(const PointXYZ p)
	{
		count += 1;
		centroid += p;

		points.push_back(p);
		return;
	}

	// Get the point, features, and labels closest to the centroid
	pair<PointXYZ, pair<vector<float>, vector<int>>> closest_point_data()
	{
		PointXYZ avg = centroid * (1.0 / count);  // 计算质心
		float min_dist = numeric_limits<float>::max();
		size_t closest_idx = 0;

		for (size_t i = 0; i < points.size(); ++i)
		{
			float dist = (points[i].x - avg.x) * (points[i].x - avg.x) +
             		(points[i].y - avg.y) * (points[i].y - avg.y) +
             		(points[i].z - avg.z) * (points[i].z - avg.z);
			if (dist < min_dist)
			{
				min_dist = dist;
				closest_idx = i;
			}
		}

		// 返回距离质心最近的点及其对应的特征和类别
		return {points[closest_idx], {points_features[closest_idx], points_labels[closest_idx]}};
	}
};



void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose);

// 实现暂时有问题
// void octree_subsampling(vector<PointXYZ>& original_points,
//                         vector<PointXYZ>& subsampled_points,
//                         vector<float>& original_features,
//                         vector<float>& subsampled_features,
//                         vector<int>& original_classes,
//                         vector<int>& subsampled_classes,
//                         float sampleDl,
//                         int verbose); 

void farthest_point_sampling(vector<PointXYZ>& original_points,
                        vector<PointXYZ>& subsampled_points,
                        vector<float>& original_features,
                        vector<float>& subsampled_features,
                        vector<int>& original_classes,
                        vector<int>& subsampled_classes,
                        size_t num_samples,
                        int verbose); 


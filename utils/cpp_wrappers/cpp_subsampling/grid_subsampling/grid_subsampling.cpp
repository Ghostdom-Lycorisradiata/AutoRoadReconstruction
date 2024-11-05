
#include "grid_subsampling.h"
#include <queue>

void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      float sampleDl,
                      int verbose) {

	// Initiate variables
	// *****************
	// Number of points in the cloud
	size_t N = original_points.size();
	size_t num_samples = static_cast<size_t>(N * 0.1);

	// Dimension of the features
	size_t fdim = original_features.size() / N;
	size_t ldim = original_classes.size() / N;

	// Limits of the cloud
	PointXYZ minCorner = min_point(original_points);
	PointXYZ maxCorner = max_point(original_points);
	PointXYZ originCorner = floor(minCorner * (1/sampleDl)) * sampleDl;

	// Dimensions of the grid
	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1;
	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1;
	//size_t sampleNZ = (size_t)floor((maxCorner.z - originCorner.z) / sampleDl) + 1;

	// Check if features and classes need to be processed
	bool use_feature = original_features.size() > 0;
	bool use_classes = original_classes.size() > 0;

	// Create the sampled map
	// **********************
	// Verbose parameters
	int i = 0;
	int nDisp = N / 100;

	// Initiate variables
	size_t iX, iY, iZ, mapIdx;
	unordered_map<size_t, SampledData> data;

	for (auto& p : original_points)
	{
		// Position of point in sample map
		iX = (size_t)floor((p.x - originCorner.x) / sampleDl);
		iY = (size_t)floor((p.y - originCorner.y) / sampleDl);
		iZ = (size_t)floor((p.z - originCorner.z) / sampleDl);
		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;

		// If not already created, create key
		if (data.count(mapIdx) < 1)
			data.emplace(mapIdx, SampledData(fdim, ldim));

		// Fill the sample map
		if (use_feature && use_classes)
			data[mapIdx].update_all(p, original_features.begin() + i * fdim, original_classes.begin() + i * ldim, fdim, ldim);
		else if (use_feature)
			data[mapIdx].update_features(p, original_features.begin() + i * fdim, fdim, ldim);
		else if (use_classes)
			data[mapIdx].update_classes(p, original_classes.begin() + i * ldim, fdim, ldim);
		else
			data[mapIdx].update_points(p);

		// Display
		i++;
		if (verbose > 1 && i%nDisp == 0)
			cout << "\rSampled Map : " << setw(3) << i / nDisp << "%";

	}

	vector<PointXYZ> temp_subsampled_points;
	vector<float> temp_subsampled_features;
	vector<int> temp_subsampled_classes;

	// Divide for barycentre and transfer to a vector
	temp_subsampled_points.reserve(data.size());
	if (use_feature)
		temp_subsampled_features.reserve(data.size() * fdim);
	if (use_classes)
		temp_subsampled_classes.reserve(data.size() * ldim);

	for (auto& v : data)
	{
		auto [closest_point, closest_data] = v.second.closest_point_data();
		subsampled_points.push_back(closest_point);
		
		if (use_feature)
		{
			subsampled_features.insert(subsampled_features.end(), closest_data.first.begin(), closest_data.first.end());
		}

		if (use_classes)
		{
			subsampled_classes.insert(subsampled_classes.end(), closest_data.second.begin(), closest_data.second.end());
		}
	}

	// farthest_point_sampling(temp_subsampled_points,
    //                  subsampled_points,
    //                  temp_subsampled_features,
    //                  subsampled_features,
    //                  temp_subsampled_classes,
    //                  subsampled_classes,
    //                  num_samples,
    //                  verbose);

	return;
}

float distance(const PointXYZ& a, const PointXYZ& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
}

void farthest_point_sampling(vector<PointXYZ>& original_points,
                        vector<PointXYZ>& subsampled_points,
                        vector<float>& original_features,
                        vector<float>& subsampled_features,
                        vector<int>& original_classes,
                        vector<int>& subsampled_classes,
                        size_t num_samples,
                        int verbose){
    
    size_t N = original_points.size();
    if (num_samples == 0) return; // 如果目标点数为0，直接返回

    // 选择第一个点作为初始点
    subsampled_points.push_back(original_points[0]);
    subsampled_classes.push_back(original_classes[0]);
    subsampled_features.insert(subsampled_features.end(),
                                original_features.begin(),
                                original_features.begin() + original_features.size() / original_points.size());

    std::set<int> sampled_indices;
    sampled_indices.insert(0); // 记录已采样的点索引

    while (subsampled_points.size() < num_samples) {
        std::vector<float> distances(original_points.size(), std::numeric_limits<float>::max());
        
        // 计算每个未采样点到已采样点的最小距离
        for (const int& sampled_index : sampled_indices) {
            for (size_t i = 0; i < original_points.size(); ++i) {
                if (sampled_indices.find(i) == sampled_indices.end()) { // 未采样的点
                    float dist = distance(original_points[i], original_points[sampled_index]);
                    distances[i] = std::min(distances[i], dist);
                }
            }
        }

        // 找到距离最远的点
        int farthest_index = std::distance(distances.begin(), std::max_element(distances.begin(), distances.end()));
        
        // 将选中的点添加到结果中
        subsampled_points.push_back(original_points[farthest_index]);
        subsampled_classes.push_back(original_classes[farthest_index]);
        subsampled_features.insert(subsampled_features.end(),
                                   original_features.begin() + farthest_index * (original_features.size() / original_points.size()),
                                   original_features.begin() + (farthest_index + 1) * (original_features.size() / original_points.size()));
        
        sampled_indices.insert(farthest_index);
    }
}

// 八叉树采样函数
// 修改后的八叉树节点类
// class OctreeNode
// {
// public:
//     // 八叉树的八个子节点
//     vector<OctreeNode*> children;
//     vector<PointXYZ> points;  // 存储该节点的点
//     vector<vector<float>> features;
//     vector<int> labels;

//     PointXYZ minCorner;  // 当前节点的最小角点
//     PointXYZ maxCorner;  // 当前节点的最大角点

//     // 构造函数
//     OctreeNode(PointXYZ minCorner, PointXYZ maxCorner) : minCorner(minCorner), maxCorner(maxCorner) {
//         children.resize(8, nullptr);
//     }

//     // 判断是否是叶节点
//     bool isLeaf() const {
//         return children[0] == nullptr;
//     }

//     // 将点添加到当前节点
//     void addPoint(const PointXYZ& point, const vector<float>& feature, int label) {
// 		if (contains(point)) {  // 确保点在当前节点范围内
// 			points.push_back(point);
// 			features.push_back(feature);
// 			labels.push_back(label);
// 		}
//     }

//     // 细分节点，将其划分为8个子节点
//     void subdivide() {
//         PointXYZ center = (minCorner + maxCorner) * 0.5;
//         PointXYZ halfSize = (maxCorner - minCorner) * 0.5;

//         // 创建8个子节点
//         for (int i = 0; i < 8; ++i) {
//             PointXYZ childMin = minCorner;
//             PointXYZ childMax = maxCorner;

//             if (i & 1) childMin.x = center.x; else childMax.x = center.x;
//             if (i & 2) childMin.y = center.y; else childMax.y = center.y;
//             if (i & 4) childMin.z = center.z; else childMax.z = center.z;

//             children[i] = new OctreeNode(childMin, childMax);
//         }
//     }

//     // 判断点是否在当前节点的范围内
//     bool contains(const PointXYZ& point) const {
//         return (point.x >= minCorner.x && point.x <= maxCorner.x) &&
//                (point.y >= minCorner.y && point.y <= maxCorner.y) &&
//                (point.z >= minCorner.z && point.z <= maxCorner.z);
//     }
// };

// 八叉树采样函数
// void octree_subsampling(vector<PointXYZ>& original_points,
//                         vector<PointXYZ>& subsampled_points,
//                         vector<float>& original_features,
//                         vector<float>& subsampled_features,
//                         vector<int>& original_classes,
//                         vector<int>& subsampled_classes,
//                         float sampleDl,
//                         int verbose) {
    
// 	// 创建根节点，假设整个点云的包围盒
//     PointXYZ min_pt = min_point(original_points);
//     PointXYZ max_pt = max_point(original_points);
//     OctreeNode* root = new OctreeNode(min_pt, max_pt);

//     // 将所有点插入八叉树中
//     size_t N = original_points.size();
//     size_t fdim = original_features.size() / N;
//     size_t ldim = original_classes.size() / N;

//     for (size_t i = 0; i < N; ++i) {
//         PointXYZ point = original_points[i];
//         vector<float> feature(original_features.begin() + i * fdim, original_features.begin() + (i + 1) * fdim);
//         int label = original_classes[i];

//         OctreeNode* node = root;
//         while (node->maxCorner.x - node->minCorner.x > sampleDl || 
//                node->maxCorner.y - node->minCorner.y > sampleDl ||
//                node->maxCorner.z - node->minCorner.z > sampleDl) {
//             if (node->isLeaf() && node->points.size() > 0) {
//                 node->subdivide();
//             }
//             int index = 0;
//             if (point.x > node->minCorner.x + (node->maxCorner.x - node->minCorner.x) / 2) index |= 1;
//             if (point.y > node->minCorner.y + (node->maxCorner.y - node->minCorner.y) / 2) index |= 2;
//             if (point.z > node->minCorner.z + (node->maxCorner.z - node->minCorner.z) / 2) index |= 4;
// 			if (node->children[index] = nullptr)
// 			{
// 				break;
// 			}
//             node = node->children[index];
//         }
//         node->addPoint(point, feature, label);
//     }

//     // 遍历八叉树，计算每个叶节点的质心
//     queue<OctreeNode*> q;
//     q.push(root);
    
//     while (!q.empty()) {
//         OctreeNode* node = q.front();
//         q.pop();

//         if (node->isLeaf()) {
//             if (node->points.size() > 0) {				
// 				// 计算质心，选择离质心最近的点作为特征，不破坏原有特征
// 				PointXYZ barycenter(0, 0, 0);
// 				for (const auto& pt : node->points) {
// 					barycenter += pt;
// 				}
// 				barycenter *= (1.0 / node->points.size());

// 				// 找到距离质心最近的点
// 				float min_distance = std::numeric_limits<float>::max();
// 				size_t closest_index = 0;

// 				for (size_t i = 0; i < node->points.size(); ++i) {
// 					float distance = std::sqrt(
// 						std::pow(node->points[i].x - barycenter.x, 2) +
// 						std::pow(node->points[i].y - barycenter.y, 2) +
// 						std::pow(node->points[i].z - barycenter.z, 2)
// 					);
// 					if (distance < min_distance) {
// 						min_distance = distance;
// 						closest_index = i;
// 					}
// 				}

// 				// 添加到结果
// 				subsampled_points.push_back(node->points[closest_index]);
// 				subsampled_features.insert(subsampled_features.end(), node->features[closest_index].begin(), node->features[closest_index].end());
// 				subsampled_classes.push_back(node->labels[closest_index]);
//             }
//         } else {
//             for (auto& child : node->children) {
//                 if (child != nullptr) {
//                     q.push(child);
//                 }
//             }
//         }
//     }

//     // 释放八叉树内存
//     delete root;
// }

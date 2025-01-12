#include "magsac_python.hpp"
#include "magsac.h"
#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "types.h"
#include "model.h"
#include "utils.h"
#include "estimators.h"
#include "most_similar_inlier_selector.h"
#include "samplers/progressive_napsac_sampler.h"
#include "samplers/probabilistic_reordering_sampler.h"
#include "samplers/importance_sampler.h"
#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/gumbel1.h"
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>


void optimizeEssentialMatrix_(
std::vector<double> &correspondences,
//std::vector<double>& dstPts,
std::vector<double>& src_K,
std::vector<double>& dst_K,
std::vector<size_t>& inliers_,
std::vector<double>& best_model,
std::vector<double>&E, double threshold, double estimated_score
)

{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);
	int iterations = 0;
	
	gcransac::EssentialMatrix model;
	model.descriptor.resize(3,3);
	//model.descriptor = estimated_model;
	//std::cout<<"points"<<points<<std::endl;
	//Eigen::Matrix3d estimated_model;
	for (size_t i = 0; i < 3; ++i) {
		for (size_t j = 0; j < 3;++j) {
			model.descriptor(j, i) = best_model[i * 3 + j];
			std::cout<<model.descriptor(j, i)<<std::endl;
		}
	}
	
	std::vector<gcransac::Model> models = {model};
	
	//std::cout<<"input best modddddddel sizzze"<<models.size()<<std::endl;

	Eigen::Matrix3d intrinsics_src,
		intrinsics_dst;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			intrinsics_src(i, j) = src_K[i * 3 + j];
		}
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			intrinsics_dst(i, j) = dst_K[i * 3 + j];
		}
	}

	gcransac::utils::DefaultEssentialMatrixEstimator estimator(intrinsics_src,
		intrinsics_dst);
	//fprintf(inliers_);
	//std::cout<<"inliers in cpp:"<<std::endl;
	//for (int    i=0;i<inliers_.size();i++)
	//{
	
	 //  std::cout<<inliers_[i]<<std::endl;
	//}
	//std::cout<<"inliers end"<<std::endl;
		
	gcransac::estimator::solver::EssentialMatrixBundleAdjustmentSolver bundleOptimizer;
	bundleOptimizer.estimateModel(
		points,
		&inliers_[0],
		inliers_.size(),
		models);
	
	//std::cout<<"models after BA:"<<models.size()<<std::endl;
	//fprintf(inliers_.size());
	const double &fx1 = intrinsics_src(0, 0);
	const double &fy1 = intrinsics_src(1, 1);
	const double &fx2 = intrinsics_dst(0, 0);
	const double &fy2 = intrinsics_dst(1, 1);

	const double threshold_normalizer = (fx1 + fx2 + fy1 + fy2) / 4.0;
	//std::cout<<fx1<<fx2<<fy1<<fy2<<std::endl;
	// The truncated least-squares threshold
	const double truncated_threshold =  3.0/2.0 *threshold / threshold_normalizer;
	// The squared least-squares threshold
	const double squared_truncated_threshold = truncated_threshold * truncated_threshold;
	//std::cout<<"threshold:"<<threshold_normalizer<<threshold<<squared_truncated_threshold<<std::endl;
	// Scoring function
	gcransac::MSACScoringFunction<gcransac::utils::DefaultEssentialMatrixEstimator> scoring;
	scoring.initialize(squared_truncated_threshold, points.rows);
	// Score of the new model
	gcransac::Score score;
	// Inliers of the new model
	std::vector<size_t> inliers;
	
	//gcransac::EssentialMatrix model;
	//mo//del.descriptor.resize(3,3);
	//model.descriptor = estimated_model;
	
	//model.descriptor = Eigen::Matrix3d::Identity();
	// Select the best model and update the inliers
	for (auto& tmp_model : models)
	{
		inliers.clear();
		//std::cout<<points<<std::endl;
		//std::cout<<"model"<<tmp_model.descriptor<<std::endl;
		score = scoring.getScore(points, // All points
		tmp_model,// The current model parameters
		estimator,// The estimator 
		squared_truncated_threshold, // The current threshold
		inliers); // The current inlier set
		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		std::cout<<"check"<<std::endl;
		//		std::cout<<"tmp model "<<tmp_model.descriptor(i, j)<<std::endl;
		//		}
		//	}
		
		//std::cout<<"score"<<score.value<<std::endl;
		//std::cout<<"estimated_score"<<score.value<<std::endl;
		//std::cout<<"next model"<<std::endl;
		// Check if the updated model is better than the best so far
		if (estimated_score < score.value)
		{
			//std::cout<<"find a better"<<std::endl;
			model.descriptor = tmp_model.descriptor;
			estimated_score = score.value;
			inliers_.swap(inliers);
		}
		
		
	}
	
	inliers.resize(num_tents);

	const int num_inliers = inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[inliers[pt_idx]] = 1;
	}

	E.resize(9);
	//ssstd::cout<<"E before filling"<<E<<std::endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			//std::cout<<"from model to E"<<i * 3 + j<<model.descriptor(i, j)<<std::endl;
			E[i * 3 + j] = (double)model.descriptor(i, j);
			//std::cout<<"E"<<i * 3 + j<<E[i * 3 + j]<<std::endl;
		}
	}
	
	//std::cout<<"_____"<<std::endl;
	
}

int findFundamentalMatrix_(std::vector<double>& srcPts,
    std::vector<double>& dstPts,
    std::vector<bool>& inliers,
    std::vector<double>& F,
    std::vector<size_t> &minimal_samples,
    std::vector<double> &point_probabilities,
    double variance,
    double sourceImageWidth,
    double sourceImageHeight,
    double destinationImageWidth,
    double destinationImageHeight,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int max_iters,
    int partition_num,
    int sampler_id,
    bool save_minimal_samples)
{
    magsac::utils::DefaultFundamentalMatrixEstimator estimator(0.1); // The robust homography estimator class containing the
    gcransac::FundamentalMatrix model; // The estimated model

    MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>* magsac;
    if (use_magsac_plus_plus)
        magsac = new MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>(
            MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>::MAGSAC_PLUS_PLUS);
    else
        magsac = new MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>(
            MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator>::MAGSAC_ORIGINAL);

    // Set the corresponding flag if the samples should be saved
    if (save_minimal_samples)
        magsac->setSampleSavingFlag(save_minimal_samples);

    magsac->setMaximumThreshold(sigma_max); // The maximum noise scale sigma allowed
    magsac->setCoreNumber(1); // The number of cores used to speed up sigma-consensus
    magsac->setPartitionNumber(partition_num); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
    magsac->setIterationLimit(max_iters);
    magsac->setMinimumIterationNumber(1000);

    int num_tents = srcPts.size() / 2;
    cv::Mat points(num_tents, 4, CV_64F);
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2 * i];
        points.at<double>(i, 1) = srcPts[2 * i + 1];
        points.at<double>(i, 2) = dstPts[2 * i];
        points.at<double>(i, 3) = dstPts[2 * i + 1];
    }

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
            { static_cast<double>(sourceImageWidth), // The width of the source image
                static_cast<double>(sourceImageHeight), // The height of the source image
                static_cast<double>(destinationImageWidth), // The width of the destination image
                static_cast<double>(destinationImageHeight) },  // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else if (sampler_id == 3)
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::GumbelSoftmaxSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4)
    {
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProbabilisticReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            variance));
    }
    else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

    ModelScore score;
    int iteration_number = 0;
    bool success = magsac->run(points, // The data points
        conf, // The required confidence in the results
        estimator, // The used estimator
        *main_sampler.get(), // The sampler used for selecting minimal samples in each iteration
        model, // The estimated model
        iteration_number, // The number of iterations
        score); // The score of the estimated model
    inliers.resize(num_tents);

    // Delete the sampler
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

    if (!success) {
        for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
            inliers[pt_idx] = false;
        }
        F.resize(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                F[i * 3 + j] = 0;
            }
        }
        return 0;
    }
    
    // Save the samples
    if (save_minimal_samples)
    {
        minimal_samples.reserve(iteration_number * 7);
        for (const auto &sample : magsac->getMinimalSamples())
            for (size_t point_idx = 0; point_idx < 7; ++point_idx)
                minimal_samples.emplace_back(sample[point_idx]);
    }

    int num_inliers = 0;
    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
        const int is_inlier = estimator.residual(points.row(pt_idx), model.descriptor) <= sigma_max;
        inliers[pt_idx] = (bool)is_inlier;
        num_inliers += is_inlier;
    }

    F.resize(9);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            F[i * 3 + j] = (double)model.descriptor(i, j);
        }
    }
    return num_inliers;
}

int findEssentialMatrix_(std::vector<double>& srcPts,
    std::vector<double>& dstPts,
    std::vector<bool>& inliers,
    std::vector<double>& E,
    std::vector<double>& src_K,
    std::vector<double>& dst_K,
    std::vector<size_t> &minimal_samples,
    std::vector<double> &point_probabilities,
    double variance,
    double sourceImageWidth,
    double sourceImageHeight,
    double destinationImageWidth,
    double destinationImageHeight,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int max_iters,
    int partition_num,
    int sampler_id,
    bool save_minimal_samples)
{
    int num_tents = srcPts.size() / 2;
    cv::Mat points(num_tents, 4, CV_64F);
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2 * i];
        points.at<double>(i, 1) = srcPts[2 * i + 1];
        points.at<double>(i, 2) = dstPts[2 * i];
        points.at<double>(i, 3) = dstPts[2 * i + 1];
    }

    Eigen::Matrix3d intrinsics_src,
        intrinsics_dst;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intrinsics_src(i, j) = src_K[i * 3 + j];
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intrinsics_dst(i, j) = dst_K[i * 3 + j];
        }
    }

    const double &fx1 = intrinsics_src(0, 0);
    const double &fy1 = intrinsics_src(1, 1);
    const double &fx2 = intrinsics_dst(0, 0);
    const double &fy2 = intrinsics_dst(1, 1);

    const double threshold_normalizer =
        (fx1 + fx2 + fy1 + fy2) / 4.0;
    const double normalized_sigma_max =
        sigma_max / threshold_normalizer;

    //cv::Mat normalized_points(points.size(), CV_64F);
    //gcransac::utils::normalizeCorrespondences(points,
     //   intrinsics_src,
     //   intrinsics_dst,
     //   normalized_points);

    magsac::utils::DefaultEssentialMatrixEstimator estimator(
        intrinsics_src,
        intrinsics_dst); // The robust essential matrix estimator class
    gcransac::EssentialMatrix model; // The estimated model

    MAGSAC<cv::Mat, magsac::utils::DefaultEssentialMatrixEstimator> magsac(
        use_magsac_plus_plus ?
            MAGSAC<cv::Mat, magsac::utils::DefaultEssentialMatrixEstimator>::MAGSAC_PLUS_PLUS : 
            MAGSAC<cv::Mat, magsac::utils::DefaultEssentialMatrixEstimator>::MAGSAC_ORIGINAL);

    // Set the corresponding flag if the samples should be saved
    if (save_minimal_samples)
        magsac.setSampleSavingFlag(save_minimal_samples);

    magsac.setMaximumThreshold(normalized_sigma_max); // The maximum noise scale sigma allowed
    magsac.setCoreNumber(1); // The number of cores used to speed up sigma-consensus
    magsac.setPartitionNumber(partition_num); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
    magsac.setIterationLimit(max_iters);
    magsac.setReferenceThreshold(magsac.getReferenceThreshold() / threshold_normalizer); // The reference threshold inside MAGSAC++ should also be normalized.
    magsac.setMinimumIterationNumber(1000);

	// Initialize the sampler used for selecting minimal samples
	// The main sampler is used for sampling in the main RANSAC loop
	typedef gcransac::sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
            { static_cast<double>(sourceImageWidth), // The width of the source image
                static_cast<double>(sourceImageHeight), // The height of the source image
                static_cast<double>(destinationImageWidth), // The width of the destination image
                static_cast<double>(destinationImageHeight) },  // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else if (sampler_id == 3)
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::GumbelSoftmaxSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4)
    {
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ProbabilisticReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            variance));
    } else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

    ModelScore score;
    int iteration_number = 0;
    bool success = magsac.run(points, // The data points
        conf, // The required confidence in the results
        estimator, // The used estimator
        *main_sampler.get(), // The sampler used for selecting minimal samples in each iteration
        model, // The estimated model
        iteration_number, // The number of iterations
        score); // The score of the estimated model
    inliers.resize(num_tents);

    // Delete the sampler
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

    if (!success) {
        for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
            inliers[pt_idx] = false;
        }
        E.resize(9);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                E[i * 3 + j] = 0;
            }
        }
        return 0;
    }
    
    // Save the samples
    if (save_minimal_samples)
    {
        minimal_samples.reserve(iteration_number * 5);
        for (const auto &sample : magsac.getMinimalSamples())
            for (size_t point_idx = 0; point_idx < 5; ++point_idx)
                minimal_samples.emplace_back(sample[point_idx]);
    }

    int num_inliers = 0;
    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
        const int is_inlier = 
            estimator.residual(points.row(pt_idx), model.descriptor) <= normalized_sigma_max;
        inliers[pt_idx] = (bool)is_inlier;
        num_inliers += is_inlier;
    }

    E.resize(9);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            E[i * 3 + j] = (double)model.descriptor(i, j);
        }
    }

    return num_inliers;
}

int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    double sourceImageWidth,
                    double sourceImageHeight,
                    double destinationImageWidth,
                    double destinationImageHeight,
					bool use_magsac_plus_plus,
                    double sigma_max,
                    double conf,
                    int max_iters,
                    int partition_num)

{
    magsac::utils::DefaultHomographyEstimator estimator; // The robust homography estimator class containing the function for the fitting and residual calculation
    gcransac::Homography model; // The estimated model

    MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>* magsac;
    if (use_magsac_plus_plus)
        magsac = new MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>(
            MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>::MAGSAC_PLUS_PLUS);
    else
        magsac = new MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>(
            MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>::MAGSAC_ORIGINAL);

    magsac->setMaximumThreshold(sigma_max); // The maximum noise scale sigma allowed
    magsac->setCoreNumber(1); // The number of cores used to speed up sigma-consensus
    magsac->setPartitionNumber(partition_num); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
    magsac->setIterationLimit(max_iters);
    
	ModelScore score;
	
    int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 4, CV_64F);
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = dstPts[2*i];
        points.at<double>(i, 3) = dstPts[2*i + 1];
    }
	
	// Initialize the sampler used for selecting minimal samples
	gcransac::sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		{ static_cast<double>(sourceImageWidth), // The width of the source image
			static_cast<double>(sourceImageHeight), // The height of the source image
			static_cast<double>(destinationImageWidth), // The width of the destination image
			static_cast<double>(destinationImageHeight) },  // The height of the destination image
		0.5); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

    bool success = magsac->run(points, // The data points
                              conf, // The required confidence in the results
                              estimator, // The used estimator
                              main_sampler, // The sampler used for selecting minimal samples in each iteration
                              model, // The estimated model
                              max_iters, // The number of iterations
							  score); // The score of the estimated model
    inliers.resize(num_tents);
    if (!success) {
        for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
            inliers[pt_idx] = false;
        }
        H.resize(9);
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                H[i*3+j] = 0;
            }
        }
        return 0;
    }

    int num_inliers = 0;
    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
        const int is_inlier = sqrt(estimator.residual(points.row(pt_idx), model.descriptor)) <= sigma_max;
        inliers[pt_idx] = (bool)is_inlier;
        num_inliers+=is_inlier;
    }
    
    H.resize(9);
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            H[i*3+j] = (double)model.descriptor(i,j);
        }
    }
    return num_inliers;
}

int adaptiveInlierSelection_(
    const std::vector<double>& srcPts_,
    const std::vector<double>& dstPts_,
    const std::vector<double>& model_,
    std::vector<bool>& inliers_,
    double &bestThreshold_,
    int problemType_,
    double maximumThreshold_,
    int minimumInlierNumber_)
{
    if (problemType_ > 2)
    {
        LOG(ERROR) << "The valid settings for variable 'problemType' are\n\t 0 (homography)\n\t 1 (fundamental matrix) \n\t 2 (essential matrix)\n";
        return 0;
    }

    int num_tents = srcPts_.size() / 2;
    cv::Mat points(num_tents, 4, CV_64F);
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts_[2 * i];
        points.at<double>(i, 1) = srcPts_[2 * i + 1];
        points.at<double>(i, 2) = dstPts_[2 * i];
        points.at<double>(i, 3) = dstPts_[2 * i + 1];
    }

    std::vector<size_t> selectedInliers;
    double bestThreshold;

    if (problemType_ == 0)
    {
        gcransac::Model model;
        model.descriptor.resize(3, 3);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 3; ++c)
                model.descriptor(r, c) = model_[3 * r + c];

        MostSimilarInlierSelector<magsac::utils::DefaultHomographyEstimator>
            inlierSelector(
                MAX(magsac::utils::DefaultHomographyEstimator::sampleSize() + 1, minimumInlierNumber_),
                maximumThreshold_);

        // The robust homography estimator class containing the function for the fitting and residual calculation
        magsac::utils::DefaultHomographyEstimator homographyEstimator;

        inlierSelector.selectInliers(points,
            homographyEstimator,
            model,
            selectedInliers,
            bestThreshold_);
    }
    else if (problemType_ == 1)
    {
        gcransac::Model model;
        model.descriptor.resize(3, 3);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 3; ++c)
                model.descriptor(r, c) = model_[3 * r + c];

        MostSimilarInlierSelector<magsac::utils::DefaultFundamentalMatrixEstimator>
            inlierSelector(
                MAX(magsac::utils::DefaultFundamentalMatrixEstimator::sampleSize() + 1, minimumInlierNumber_),
                maximumThreshold_);

        // The robust fundamental matrix estimator class containing the function for the fitting and residual calculation
        magsac::utils::DefaultFundamentalMatrixEstimator fundamentalEstimator(maximumThreshold_);

        inlierSelector.selectInliers(points,
            fundamentalEstimator,
            model,
            selectedInliers,
            bestThreshold_);
    }
    else
    {
        LOG(INFO) << "Note: for essential matrices, the correspondences should be normalized by the intrinsic camera matrices.";

        gcransac::Model model;
        model.descriptor.resize(3, 3);
        for (size_t r = 0; r < 3; ++r)
            for (size_t c = 0; c < 3; ++c)
                model.descriptor(r, c) = model_[3 * r + c];

        MostSimilarInlierSelector<magsac::utils::DefaultEssentialMatrixEstimator>
            inlierSelector(
                MAX(magsac::utils::DefaultEssentialMatrixEstimator::sampleSize() + 1, minimumInlierNumber_),
                maximumThreshold_);

        // The robust essential matrix estimator class containing the function for the fitting and residual calculation
        magsac::utils::DefaultEssentialMatrixEstimator essentialEstimator(
            Eigen::Matrix3d::Identity(),
            Eigen::Matrix3d::Identity());

        inlierSelector.selectInliers(points,
            essentialEstimator,
            model,
            selectedInliers,
            bestThreshold_);
    }

    inliers_.resize(num_tents);

    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx)
        inliers_[pt_idx] = false;
   
    for (const auto& inlierIdx : selectedInliers)
        inliers_[inlierIdx] = true;

    return selectedInliers.size();
}

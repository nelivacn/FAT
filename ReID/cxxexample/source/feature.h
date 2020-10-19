/*!
 * Copyright (c) 2020 by NELIVA
 * \file common.hpp
 * \brief
 * \author Jia Guo
*/
#ifndef FAT_FEATURE_H_
#define FAT_FEATURE_H_
#include "common.hpp"
#include <opencv2/opencv.hpp>

namespace FAT {

class Feature {
public:
    Feature(const std::string& prototxt, const std::string& model_file);
    static int FeatureLength() { return 2048; }
    void Get(const cv::Mat& img, const Face& face, float* feat);


private:
    cv::dnn::Net net_;
    cv::Mat dst_;
 
};

}

#endif



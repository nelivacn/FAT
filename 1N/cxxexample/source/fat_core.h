/*!
 * Copyright (c) 2020 by NELIVA
 * \file common.hpp
 * \brief
 * \author Jia Guo
*/
#ifndef FAT_CORE_H_
#define FAT_CORE_H_

#include <stdio.h>
#include <cmath>
#include <algorithm>  
#include <string>
#include <iostream>
#include <fstream>
#include "detect.h"
#include "feature.h"
#include "opencv2/core/cuda.hpp"


namespace FAT {

struct Image
{
    int nC;
    int nW;
    int nH;
    char* pData;
};

class FATImpl {
    public:
        FATImpl();
        virtual ~FATImpl();

        static int FeatureLength() { return Feature::FeatureLength(); }

        void Load(const std::string& rdir);

        void GetFeature(char* img, int width, int height, int channel, int im_type, float* feat);
        // int PredictMulti(const std::vector<Image> &vImage, std::vector<float> &vFeatureStream);
        void GetFeatures(const std::vector<Image> &images, std::vector<float*> &features);
        // void Finalize(float* gallery_feats, int* gallery_labels, int N, int K);
        // void GetTopK(float* im_feat, int* topk, float* sim);
        // void GetTopKs(const std::vector<float*> &im_feats, std::vector<int*> &topks, std::vector<float*> &sims);


    private:
        std::shared_ptr<Detector> detector_;
        std::shared_ptr<Feature> feature_;
        // float* gallery_feats_;
        // cv::Mat gallery_feats_;
        // cv::cuda::GpuMat gallery_feats_;
        // int* gallery_labels_;
        // int N_;
        // int K_;
        // std::vector<std::pair<float, int> > buffer_;

};




class FATGalleryQuery{

public:
    FATGalleryQuery(): gallery_labels_(NULL), N_(0) {}
    virtual ~FATGalleryQuery(){}
    static int FeatureLength() { return Feature::FeatureLength(); }
    void Finalize(float* gallery_feats, int* gallery_labels, int N, int K);
    void GetTopK(float* im_feat, int* topk, float* sim);
    void GetTopKs(const std::vector<float*> &im_feats, std::vector<int*> &topks, std::vector<float*> &sims);
private:
    cv::cuda::GpuMat gallery_feats_;
    int* gallery_labels_;
    int N_;
    int K_;
    std::vector<std::pair<float, int> > buffer_;

};

}

#endif

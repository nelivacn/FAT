/*!
 * Copyright (c) 2020 by NELIVA
 * \file fat_core.cc
 * \brief
 * \author Jia Guo
*/

#include <cmath>
#include "fat_core.h"
using namespace FAT;

FATImpl::FATImpl()
: gallery_labels_(NULL), N_(0) 
{
}
FATImpl::~FATImpl() {
}


void FATImpl::Load(const std::string& rdir) {
    cv::setNumThreads(1);
    std::cout << "load" << std::endl;
    detector_.reset(new Detector(rdir+"/det.cfg", rdir+"/det.weights", rdir+"/coco.names", 0.3, 224));
    feature_.reset(new Feature(rdir+"/ResNet_50_test.prototxt", rdir+"/ResNet_50.caffemodel"));
}

void FATImpl::GetFeature(char* c_img, int width, int height, int channel, int im_type, float* feat)  {
    cv::Mat im;
    assert(channel==1 || channel==3);
    if(channel==1) {
        im = cv::Mat(height, width, CV_8UC1, c_img);
    }
    else {
        im = cv::Mat(height, width, CV_8UC3, c_img);
    }

    cv::Mat person = detector_->Detect(im, 0.5, 1);
    feature_->Get(person, Face(), feat);
}


void FATImpl::Finalize(float* gallery_feats, int* gallery_labels, int N, int K) {
    //gallery_feats_ = gallery_feats;
    gallery_feats_ = cv::Mat(cv::Size(FeatureLength(), N), CV_32FC1, gallery_feats);
    gallery_labels_ = gallery_labels;
    N_ = N;
    K_ = K;
    buffer_.resize(N_);
}


void FATImpl::GetTopK(float* im_feat, int* topk, float* sim) {
    int size = FeatureLength();
    cv::Mat im_feat_mat = cv::Mat(cv::Size(1, size), CV_32FC1, im_feat);
    cv::Mat ret = gallery_feats_ * im_feat_mat / 2 + .5;
    //for(int n=0;n<N_;n++) {
    //    int a = n*size;
    //    //int b = (n+1)*size;
    //    float sim = 0.0;
    //    for(int i=0;i<size;i++) {
    //        sim += gallery_feats_[a+i] * im_feat[i];
    //    }
    //    buffer_[n] = std::make_pair(sim, n);
    //}
    for(int n=0;n<N_;n++) {
        float v = ret.at<float>(n);
        buffer_[n] = std::make_pair(v, n);
    }
    std::sort(buffer_.begin(), buffer_.end(), std::greater<std::pair<float, int>>());
    for(int i=0;i<K_;i++) {
        topk[i] = buffer_[i].second;
        sim[i] = buffer_[i].first;
    }
}



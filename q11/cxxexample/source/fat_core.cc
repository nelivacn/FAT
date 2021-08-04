/*!
 * Copyright (c) 2020 by NELIVA
 * \file common.hpp
 * \brief
 * \author Jia Guo
*/

#include <cmath>
#include "fat_core.h"
using namespace FAT;

FATImpl::FATImpl() {
}
FATImpl::~FATImpl() {
}


void FATImpl::Load(const std::string& rdir) {
    cv::setNumThreads(1);
    detector_.reset(new Detector(rdir+"/det_model.prototxt", rdir+"/det_model.caffemodel", 0.3, 224));
    feature_.reset(new Feature(rdir+"/recognition.onnx"));
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
    //cv::imwrite("./test.jpg", im);
    //memcpy(dst_.data, v, 2 * 5 * sizeof(float));
    std::vector<Face> faces = detector_->Detect(im, 0.5, 1);
    if(faces.empty()) {
        faces = detector_->Detect(im, 0.05, 1);
    }
    //std::cout<<"Detected: "<<faces.size()<<std::endl;
    if(faces.empty()) {
        faces.resize(1); //append empty face item
    }
    feature_->Get(im, faces[0], feat);
}

float FATImpl::GetSim(float* im1_feat, float* im2_feat) {
    int size = FeatureLength();
    float norm1 = 0.0;
    float norm2 = 0.0;
    float sim = 0.0;
    for(int i=0;i<size;i++) {
        norm1 += im1_feat[i]*im1_feat[i];
        norm2 += im2_feat[i]*im2_feat[i];
        sim += im1_feat[i]*im2_feat[i];
    }
    sim /= sqrt(norm1+0.00001);
    sim /= sqrt(norm2+0.00001);
    sim = (sim+1.0)/2.0;
    return sim;
}


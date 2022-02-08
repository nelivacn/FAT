/*!
 * Copyright (c) 2020 by NELIVA
 * \file fat_core.cc
 * \brief
 * \author Jia Guo
*/

#include <cmath>
#include "fat_core.h"
#include "opencv2/cudaarithm.hpp"


using namespace FAT;

FATImpl::FATImpl()
{
}
FATImpl::~FATImpl() {
}


void FATImpl::Load(const std::string& rdir) {
    // cv::setNumThreads(1);
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
    // cv::imwrite("./test.jpg", im);
    //memcpy(dst_.data, v, 2 * 5 * sizeof(float));
    std::vector<Face> faces = detector_->Detect(im, 0.5, 1);
    if(faces.empty()) {
        faces = detector_->Detect(im, 0.05, 1);
    }
    //std::cout<<"Detected: "<<faces.size()<<std::endl;
    // std::cout << "faces: " << 
    //         faces[0].bbox[0] << ", " << 
    //         faces[0].bbox[1] << ", " << 
    //         faces[0].bbox[2] << ", " << 
    //         faces[0].bbox[3] << std::endl;
    if(faces.empty()) {
        faces.resize(1); //append empty face item
    }
    feature_->Get(im, faces[0], feat);
}

void FATImpl::GetFeatures(const std::vector<Image> &images, std::vector<float*> &features){
    std::vector<cv::Mat> cv_imgs;
    for(int i = 0; i < images.size(); i++){
        cv_imgs.push_back(cv::Mat(images[i].nH, images[i].nW, CV_8UC3, images[i].pData));
    }
    std::vector<std::vector<Face>> faces = detector_->DetectBatch(cv_imgs, 0.5, 1);
    // for(int i = 0; i < faces.size(); i++){
    //     std::cout << "faces: " << 
    //         faces[i][0].bbox[0] << ", " << 
    //         faces[i][0].bbox[1] << ", " << 
    //         faces[i][0].bbox[2] << ", " << 
    //         faces[i][0].bbox[3] << std::endl;
    // }
    // for(int i = 0; i < faces.size(); i++){
    //     if(faces[i].size() == 0){
    //         faces[i].resize(1);
    //     }
    // }
    // std::cout << "do gets...." << std::endl;
    feature_->Gets(cv_imgs, faces, features);
}


void FATGalleryQuery::Finalize(float* gallery_feats, int* gallery_labels, int N, int K) {
    //gallery_feats_ = gallery_feats;
    cv::cuda::setDevice(0);
    gallery_feats_.upload(cv::Mat(cv::Size(FeatureLength(), N), CV_32FC1, gallery_feats));
    gallery_labels_ = gallery_labels;
    N_ = N;
    K_ = K;
    /*std::cout << "top k: " << K_ << ", N: " << N_ << std::endl;
    for(int i = 0; i < N_; i++){
        std::cout << "[";
        for(int l = 0; l < 5; l++){
            std::cout << gallery_feats[l + i * 512] << ", ";
        }
        std::cout << "]" << std::endl;
    }*/
    buffer_.resize(N_);
}


void FATGalleryQuery::GetTopK(float* im_feat, int* topk, float* sim) {
    cv::cuda::setDevice(0);
    int size = FeatureLength();
    cv::cuda::GpuMat im_feat_mat;
    im_feat_mat.upload(cv::Mat(cv::Size(1, size), CV_32FC1, im_feat));
    cv::cuda::GpuMat gpu_ret;
    // cv::Mat cpu_ret;
    // std::cout << im_feat_mat.cols << ", " << gallery_feats_.rows << std::endl;
    cv::cuda::gemm(gallery_feats_, im_feat_mat, 1.0, cv::cuda::GpuMat(), 0.0, gpu_ret);
    // std::cout << gpu_ret.cols << ", " << gpu_ret.rows << ", " << gpu_ret.empty() << ", " << gpu_ret.isContinuous() << ", " << gpu_ret.type() << std::endl;
    cv::Mat cpu_ret; // (cv::Size(gpu_ret.cols, gpu_ret.rows), CV_32FC1, cv::Scalar(0.0f));
    gpu_ret.download(cpu_ret);
    // for(int n=0;n<N_;n++) {
    //    int a = n*size;
    //    //int b = (n+1)*size;
    //    float sim = 0.0;
    //    for(int i=0;i<size;i++) {
    //        sim += gallery_feats_[a+i] * im_feat[i];
    //    }
    //    buffer_[n] = std::make_pair(sim, n);
    // }
    for(int n=0;n<cpu_ret.rows;n++) {
        float v = cpu_ret.at<float>(n);
        buffer_[n] = std::make_pair(v, n);
    }
    std::sort(buffer_.begin(), buffer_.end(), std::greater<std::pair<float, int>>());
    for(int i=0;i<K_;i++) {
        topk[i] = buffer_[i].second;
        sim[i] = buffer_[i].first;
        // std::cout << "result " << i << ": " << topk[i] << ", " << sim[i] << std::endl;
    }
}


void FATGalleryQuery::GetTopKs(const std::vector<float*> &im_feats, std::vector<int*> &topks, std::vector<float*> &sims)
{
    assert(im_feats.size() == topks.size());
    assert(im_feats.size() == sims.size());
    cv::cuda::setDevice(0);
    int size = FeatureLength();
    cv::cuda::GpuMat im_feats_mat(cv::Mat(cv::Size(size, im_feats.size()), CV_32FC1));
    for(int i = 0; i < im_feats.size(); i ++){
        im_feats_mat(cv::Range(i,i+1), cv::Range::all()).upload(cv::Mat(cv::Size(size, 1), CV_32FC1, im_feats[i]));
    }
    
    cv::cuda::GpuMat gpu_ret;
    // cv::Mat cpu_ret;
    // std::cout << im_feat_mat.cols << ", " << gallery_feats_.rows << std::endl;
    // N X n
    cv::cuda::gemm(gallery_feats_, im_feats_mat, 1.0, cv::cuda::GpuMat(), 0.0, gpu_ret, cv::GEMM_2_T);
    // std::cout << gpu_ret.cols << ", " << gpu_ret.rows << ", " << gpu_ret.empty() << ", " << gpu_ret.isContinuous() << ", " << gpu_ret.type() << std::endl;
    cv::Mat cpu_ret; // (cv::Size(gpu_ret.cols, gpu_ret.rows), CV_32FC1, cv::Scalar(0.0f));
    gpu_ret.download(cpu_ret);
    // for(int n=0;n<N_;n++) {
    //    int a = n*size;
    //    //int b = (n+1)*size;
    //    float sim = 0.0;
    //    for(int i=0;i<size;i++) {
    //        sim += gallery_feats_[a+i] * im_feat[i];
    //    }
    //    buffer_[n] = std::make_pair(sim, n);
    // }
    for(int i = 0; i < im_feats.size(); i++){
        for(int l=0; l<cpu_ret.rows; l++) {
            float v = cpu_ret.at<float>(l, i);
            buffer_[l] = std::make_pair(v, l);
        }

        std::sort(buffer_.begin(), buffer_.end(), std::greater<std::pair<float, int>>());
        for(int k=0; k<K_; k++) {
            topks[i][k] = buffer_[k].second;
            sims[i][k] = buffer_[k].first;
            // std::cout << "result " << i << ": " << topks[i][k] << ", " << sims[i][k] << std::endl;
        }
    }
    
}

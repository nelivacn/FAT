/*!
 * Copyright (c) 2020 by NELIVA
 * \file common.hpp
 * \brief
 * \author Jia Guo
*/

#include "anchor_generator.h"
#include "detect.h"
#include <iostream>
#include <algorithm>
#include <fstream>


using namespace FAT;

Detector::Detector(const std::string& model_file,
                   const std::string& weights_file,  
                   const std::string& cls_file,                  
                   float nms_thresh,
                   std::size_t input_size)
: nms_thresh_(nms_thresh), num_channels_(3), input_size_(input_size)
{
  /* Load the network. */
    net_ = cv::dnn::readNetFromDarknet(model_file, weights_file);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "load det" << std::endl;
    std::vector<int> feat_stride_fpn = {32, 16, 8};
    std::map<int, AnchorCfg> anchor_cfg = {
        {32, AnchorCfg(std::vector<float>{32,16}, std::vector<float>{1}, 16)},
        {16, AnchorCfg(std::vector<float>{8,4}, std::vector<float>{1}, 16)},
        {8,AnchorCfg(std::vector<float>{2,1}, std::vector<float>{1}, 16)}
    };
    ac_.resize(feat_stride_fpn.size());
    for (std::size_t i = 0; i < feat_stride_fpn.size(); ++i) {
        int stride = feat_stride_fpn[i];
        ac_[i].Init(stride, anchor_cfg[stride]);
    }

    std::vector<int> outLayers = net_.getUnconnectedOutLayers();
    std::vector<cv::String> layersNames = net_.getLayerNames();
    output_names_.resize(outLayers.size());
    for(size_t i =0;i<outLayers.size();i++){
        output_names_[i] = layersNames[outLayers[i]-1];
    }

    std::string classesFile = cls_file;
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while(getline(ifs,line))classes_.push_back(line);
}

float Detector::Preprocess_(const cv::Mat& img, cv::Mat& output)
{

    cv::Mat sample, scaled;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;
    std::size_t w = img.cols;
    std::size_t h = img.rows;
    std::size_t input_height = input_size_;
    std::size_t input_width = input_size_;
    float scale = 1.0;
    float aspect_ratio = float(h)/w;
    float target_ar = float(input_height) / input_width;
    if (aspect_ratio<=target_ar)
    {
        std::size_t _w = input_width;
        std::size_t _h = static_cast<std::size_t>(input_width* aspect_ratio);
        cv::resize(sample,scaled, cv::Size(_w,_h));
        scale = float(_h)/h;
        cv::copyMakeBorder(scaled,output,0,input_height-_h,0,0,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
    }
    else{
        std::size_t _h = input_height;
        std::size_t _w = static_cast<std::size_t>(input_height /aspect_ratio );
        cv::resize(sample,scaled,cv::Size(_w,_h));
        scale = float(_w)/w;
        cv::copyMakeBorder(scaled,output,0,0,0,input_width-_w,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
    }
    return scale;
}


cv::Mat Detector::Detect(const cv::Mat& input_img, float score_thresh, std::size_t det_max)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(input_img,blob,1/255.0,cv::Size(416, 416));
    net_.setInput(blob);
    std::vector<cv::Mat> outs;
    net_.forward(outs, output_names_);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for(size_t i=0;i<outs.size();i++){
        float* data = (float*)outs[i].data;
        for(int j=0;j<outs[i].rows;j++,data+=outs[i].cols){
            cv::Mat scores = outs[i].row(j).colRange(5,outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            
            cv::minMaxLoc(scores,0,&confidence,0,&classIdPoint);
            if(confidence>0.5){
                int centerX = (int)(data[0]*input_img.cols);
                int centerY = (int)(data[1]*input_img.rows);
                int width = (int)(data[2]*input_img.cols);
                int height = (int)(data[3]*input_img.rows);
                int left = centerX-width/2;
                int top = centerY-height/2;

                if (classIdPoint.x == 0) {

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }

        }
        
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes,confidences,0.5,0.4,indices);
    for(size_t i=0;i<indices.size();i++){
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        cv::Mat result = input_img(box);
        return result;
    }
    return input_img;
}



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


using namespace FAT;

/*void printMat(const cv::Mat &image)
{
    uint8_t  *myData = image.data;
    int width = image.cols;
    int height = image.rows;
    int _stride = image.step;//in case cols != strides
    for(int i = 0; i < height; i++)
    {
	for(int j = 0; j < width; j++)
	{
	    uint8_t  val = myData[ i * _stride + j];
	    cout << val;

	    //do whatever you want with your value
	}
    }
    cout << endl;
}*/



Detector::Detector(const std::string& model_file,
                   const std::string& weights_file,                  
                   float nms_thresh,
                   std::size_t input_size)
: nms_thresh_(nms_thresh), num_channels_(3), input_size_(input_size)
{
  /* Load the network. */
    net_ = cv::dnn::readNetFromCaffe(model_file, weights_file);
    // net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    cv::cuda::setDevice(0);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    //output_names_ = {"face_rpn_cls_prob_reshape_stride32", "face_rpn_bbox_pred_stride32", "face_rpn_landmark_pred_stride32",
    //           "face_rpn_cls_prob_reshape_stride16", "face_rpn_bbox_pred_stride16", "face_rpn_landmark_pred_stride16",
    //           "face_rpn_cls_prob_reshape_stride8", "face_rpn_bbox_pred_stride8", "face_rpn_landmark_pred_stride8"};
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
    //cout<<"img1"<<img.rows<<endl;
    //cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    //if(sample.size()!=input_geometry_)
    //{
    //    ratio_w = float(img.cols)/float(input_geometry_.width);
    //    ratio_h = float(img.rows)/float(input_geometry_.height);
    //    cv::resize(sample, sample_resized, input_geometry_);
    //}
    //else
    //    sample_resized=sample;
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
    //if (num_channels_ == 3)
    //    sample_resized.convertTo(sample_float, CV_32FC3);
    //else
    //    sample_resized.convertTo(sample_float, CV_32FC1); 
    //cv::split(sample_float, *input_channels);
}


std::vector<Face> Detector::Detect(const cv::Mat& img, float score_thresh, std::size_t det_max)
{
    //cout << "input.rows :"<<input.rows << "   "<< "fd_h_ :"<<fd_h_<<endl;
    //cout<<"input.cols :"<<input.cols <<"   "<<"fd_w_ :" << fd_w_<<endl;
    //cout<< "input.channels():"<<input.channels()<<"  "<<"fd_c_: "<<fd_c_<<endl;
    //assert(input.rows == fd_h_ && input.cols == fd_w_ && input.channels() == fd_c_);
   
    cv::Mat input_img;
    float im_scale = Preprocess_(img, input_img);
    auto blob = cv::dnn::blobFromImage(input_img, 1.0, cv::Size(input_size_, input_size_), cv::Scalar(0.0, 0.0, 0.0), true, false);
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, output_names_);
    std::vector<Face> proposals;
    for (std::size_t idx = 0; idx < ac_.size(); idx++) {
        cv::Mat cls = outputs[idx*3];
        cv::Mat reg = outputs[idx*3+1];
        cv::Mat pts = outputs[idx*3+2];
        ac_[idx].Predict(cls, reg, pts, score_thresh, proposals);
    }

    // nms
    //std::cout<<"Size before NMS: "<<proposals.size()<<std::endl;
    std::vector<Face> result;
    NMS_CPU(proposals, nms_thresh_, result);
    //std::cout<<"Size after NMS: "<<result.size()<<std::endl;
    for(Face& face : result) {
        //for(int i=0;i<5;i++) {
        //    std::cout<<"A:"<<face.pts[i].x<<","<<face.pts[i].y<<std::endl;
        //}
        for(int i=0;i<4;i++) {
            face.bbox[i] /= im_scale;
        }
        for(std::size_t i=0;i<face.pts.size();i++) {
            face.pts[i].x /= im_scale;
            face.pts[i].y /= im_scale;
        }
        //for(int i=0;i<5;i++) {
        //    std::cout<<"B:"<<face.pts[i].x<<","<<face.pts[i].y<<std::endl;
        //}
    }

    std::vector<std::pair<float, std::size_t>> sel;
    for(std::size_t i=0;i<result.size();i++) {
        const Face& face = result[i];
        float w = face.bbox[2] - face.bbox[0];
        float h = face.bbox[3] - face.bbox[1];
        float wc = (face.bbox[2] + face.bbox[0])/2;
        float hc = (face.bbox[3] + face.bbox[1])/2;
        float wd = wc - img.cols/2;
        float hd = hc - img.rows/2;
        float area = w*h;
        float score = area - (wd*wd+hd*hd)*2.0;
        score *= -1.0;
        sel.push_back(std::make_pair(score, i));
    }
    std::sort(sel.begin(), sel.end());
    std::size_t ret_count = sel.size();
    if(det_max>0 and ret_count>det_max) ret_count = det_max;
    std::vector<Face> r2;
    for(std::size_t i=0;i<ret_count;i++) {
        r2.push_back( result[sel[i].second] );
    }

    //printf("final result %d\n", result.size());
    //result[0].print();
    return r2;
}



std::vector<std::vector<Face>> Detector::DetectBatch(
    const std::vector<cv::Mat> imgs, float score_thresh, std::size_t det_max){
    // do process
    std::vector<cv::Mat> input_imgs(imgs.size());
    std::vector<float> im_scales(imgs.size());
    for(int i = 0; i < imgs.size(); i++ ){
        im_scales[i] = Preprocess_(imgs[i], input_imgs[i]);
        // cv::imwrite(std::to_string(i) + ".jpg", input_imgs[i]);
    }
    auto blob = cv::dnn::blobFromImages(input_imgs, 1.0, cv::Size(input_size_, input_size_), cv::Scalar(0.0, 0.0, 0.0), true, false);
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, output_names_);
    std::vector<std::vector<Face>> _final_faces;
    // std::cout << "outputs size: " << outputs.size() << std::endl;
    for(int n = 0; n< input_imgs.size(); n++){
        std::vector<Face> proposals;
        for (std::size_t idx = 0; idx < ac_.size(); idx++) {
            std::vector<cv::Range> ranges = {cv::Range(n,n+1), cv::Range::all(), cv::Range::all(), cv::Range::all()};
            cv::Mat cls = outputs[idx*3]  (ranges);
            cv::Mat reg = outputs[idx*3+1](ranges);
            cv::Mat pts = outputs[idx*3+2](ranges);
            // ac_[idx].Predict(cls, reg, pts, score_thresh, proposals);
            // for(int n = 0; n < imgs.size(); n++){
                
            // }
            // std::cout << "cls size: " << cls.size.dims()<< ": " << cls.size[0] << " ," << cls.size[1] << ", " << cls.size[2] << ", " << cls.size[3] << std::endl;
            // std::cout << "reg size: " << reg.size.dims()<< ": " << reg.size[0] << " ," << reg.size[1] << ", " << reg.size[2] << ", " << reg.size[3] << std::endl;
            // std::cout << "pts size: " << pts.size.dims()<< ": " << pts.size[0] << " ," << pts.size[1] << ", " << pts.size[2] << ", " << pts.size[3] << std::endl;
            ac_[idx].Predict(cls, reg, pts, score_thresh, proposals);
        }
        std::vector<Face> result;
        NMS_CPU(proposals, nms_thresh_, result);  
        for(Face& face : result) {
            for(int i=0;i<4;i++) {
                face.bbox[i] /= im_scales[n];
            }
            // std::cout << "fpts size: " << face.pts.size() << std::endl;
            for(std::size_t i=0;i<face.pts.size();i++) {
                face.pts[i].x /= im_scales[n];
                face.pts[i].y /= im_scales[n];
            }
        }
        std::vector<std::pair<float, std::size_t>> sel;
        for(std::size_t i=0;i<result.size();i++) {
            const Face& face = result[i];
            float w = face.bbox[2] - face.bbox[0];
            float h = face.bbox[3] - face.bbox[1];
            float wc = (face.bbox[2] + face.bbox[0])/2;
            float hc = (face.bbox[3] + face.bbox[1])/2;
            float wd = wc - imgs[n].cols/2;
            float hd = hc - imgs[n].rows/2;
            float area = w*h;
            float score = area - (wd*wd+hd*hd)*2.0;
            score *= -1.0;
            sel.push_back(std::make_pair(score, i));
        }
        std::sort(sel.begin(), sel.end());
        std::size_t ret_count = sel.size();
        if(det_max>0 and ret_count>det_max) ret_count = det_max;
        std::vector<Face> r2;
        for(std::size_t i=0;i<ret_count;i++) {
            r2.push_back( result[sel[i].second] );
        }
        // std::cout << "faces: " << r2[0].bbox[0] << ", " << r2[0].bbox[1] << ", " << r2[0].bbox[2] << ", " << r2[0].bbox[3] << std::endl;
        _final_faces.push_back(r2);
    }
    return _final_faces;
}


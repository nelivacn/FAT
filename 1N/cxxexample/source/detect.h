/*!
 * Copyright (c) 2020 by NELIVA
 * \file common.hpp
 * \brief
 * \author Jia Guo
*/
#ifndef FAT_DETECT_H_
#define FAT_DETECT_H_
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "anchor_generator.h"

namespace FAT {

class Detector {
public:
    Detector(const std::string& model_file,
       const std::string& weights_file,          
       float nms_thresh,
       std::size_t input_size);

    std::vector<Face> Detect(const cv::Mat& img, float score_thresh, std::size_t det_max=1);
    std::vector<std::vector<Face>> DetectBatch(
        const std::vector<cv::Mat> imgs, float score_thresh, std::size_t det_max=1);
    static void NMS_CPU(std::vector<Face>& faces, float threshold, std::vector<Face>& out) {
        if(faces.size() == 0) return;

        std::vector<std::size_t> idx(faces.size());
        for(std::size_t i = 0; i < idx.size(); i++) {idx[i] = i;}

        //descending sort
        //sort(faces.begin(), faces.end(), std::greater<Face>());
        std::sort(faces.begin(), faces.end(), FaceCompare);

        while(idx.size() > 0)
        {
            int good_idx = idx[0];
            out.push_back(faces[good_idx]);

            std::vector<size_t> tmp = idx;
            //idx.clear();
            idx.resize(0);
            for(std::size_t i = 1; i < tmp.size(); i++)
            {
                int tmp_i = tmp[i];
                float inter_x1 = std::max( faces[good_idx].bbox[0], faces[tmp_i].bbox[0] );
                float inter_y1 = std::max( faces[good_idx].bbox[1], faces[tmp_i].bbox[1] );
                float inter_x2 = std::min( faces[good_idx].bbox[2], faces[tmp_i].bbox[2] );
                float inter_y2 = std::min( faces[good_idx].bbox[3], faces[tmp_i].bbox[3] );

                float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
                float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

                float inter_area = w * h;
                float area_1 = (faces[good_idx].bbox[2] - faces[good_idx].bbox[0] + 1) * (faces[good_idx].bbox[3] - faces[good_idx].bbox[1] + 1);
                float area_2 = (faces[tmp_i].bbox[2] - faces[tmp_i].bbox[0] + 1) * (faces[tmp_i].bbox[3] - faces[tmp_i].bbox[1] + 1);
                float o = inter_area / (area_1 + area_2 - inter_area);           
                if( o <= threshold )
                    idx.push_back(tmp_i);
            }
        }
    }
private:
    float Preprocess_(const cv::Mat& img, cv::Mat& output);
    static bool FaceCompare(const Face& left, const Face& right) {
        return left.score>right.score;
    }

private:
    cv::dnn::Net net_;
    float nms_thresh_;
    std::size_t num_channels_;
    std::size_t input_size_;
    std::vector<AnchorGenerator> ac_;
    std::vector<std::string> output_names_ = {"face_rpn_cls_prob_reshape_stride32", "face_rpn_bbox_pred_stride32", "face_rpn_landmark_pred_stride32",
               "face_rpn_cls_prob_reshape_stride16", "face_rpn_bbox_pred_stride16", "face_rpn_landmark_pred_stride16",
               "face_rpn_cls_prob_reshape_stride8", "face_rpn_bbox_pred_stride8", "face_rpn_landmark_pred_stride8"};
 
};

}

#endif


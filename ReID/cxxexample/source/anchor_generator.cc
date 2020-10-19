#include "anchor_generator.h"
#include <iostream>
using namespace FAT;
AnchorGenerator::AnchorGenerator() {
}

AnchorGenerator::~AnchorGenerator() {
}

// init different anchors
void AnchorGenerator::Init(int stride, const AnchorCfg& cfg) {
    CRect2f base_anchor(0, 0, cfg.BASE_SIZE-1, cfg.BASE_SIZE-1);
    std::vector<CRect2f> ratio_anchors;
    // get ratio anchors
    _ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
    _scale_enum(ratio_anchors, cfg.SCALES, preset_anchors_);

    anchor_stride_ = stride;

    anchor_num_ = preset_anchors_.size();
}

void AnchorGenerator::Predict(const cv::Mat& cls, const cv::Mat& reg, const cv::Mat& pts, float score_thresh, std::vector<Face>& result) const {


    assert(cls.size[1] == anchor_num_*2);   //anchor_num=2
    assert(reg.size[1] == anchor_num_*4);
    assert(pts.size[1] % anchor_num_ == 0);
    int pts_length = pts.size[1]/anchor_num_/2; 
    assert(pts_length==5);


    int height = cls.size[2];
    int width = cls.size[3];
    int area = height*width;

    float* cls_data = (float*)cls.data;
    float* reg_data = (float*)reg.data;
    float* pts_data = (float*)pts.data;
    for (int a = 0; a < anchor_num_; a++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                //float score = cls.at<float>(0, anchor_num_+a, h, w);
                int id = h*width+w;
                float score = cls_data[ (anchor_num_+a)*area + id ];
                if(score >= score_thresh) 
                {
                    //std::cout <<"clsData[(anchor_num + a)*step + id]: "<< clsData[(anchor_num + a)*step + id] << std::endl;
                    CRect2f box(w * anchor_stride_ + preset_anchors_[a][0],
                                h * anchor_stride_ + preset_anchors_[a][1],
                                w * anchor_stride_ + preset_anchors_[a][2],
                                h * anchor_stride_ + preset_anchors_[a][3]);

                    //printf("box::%f %f %f %f\n", box[0], box[1], box[2], box[3]);
                    CRect2f delta(reg_data[(a*4+0)*area+id],
                                  reg_data[(a*4+1)*area+id],
                                  reg_data[(a*4+2)*area+id],
                                  reg_data[(a*4+3)*area+id]);
                    Face face;
                    bbox_pred(box, delta, face.bbox);
                    face.score = score;
                    std::vector<cv::Point2f> pts_delta(pts_length);
                    for (int p = 0; p < pts_length; ++p) {
                        pts_delta[p].x = pts_data[(a*pts_length*2+p*2)*area+id];
                        pts_delta[p].y = pts_data[(a*pts_length*2+p*2+1)*area+id];
                    }                		
                    landmark_pred(box, pts_delta, face.pts);
                    //std::cout<<"Score: "<<face.score<<std::endl;
                    result.push_back(face);
                }
            }
        }
    }
}

void AnchorGenerator::_ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors) {
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    ratio_anchors.clear();
    float sz = w * h;
    for (std::size_t s = 0; s < ratios.size(); ++s) {
      	float r = ratios[s];
      	float size_ratios = sz / r;
      	float ws = std::sqrt(size_ratios);
      	float hs = ws * r;
      	ratio_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
		    y_ctr - 0.5 * (hs - 1),
		    x_ctr + 0.5 * (ws - 1),
		    y_ctr + 0.5 * (hs - 1)));
    }
}

void AnchorGenerator::_scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors) {
    scale_anchors.clear();
    for (std::size_t a = 0; a < ratio_anchor.size(); ++a) {
	CRect2f anchor = ratio_anchor[a];
	float w = anchor[2] - anchor[0] + 1;
	float h = anchor[3] - anchor[1] + 1;
	float x_ctr = anchor[0] + 0.5 * (w - 1);
	float y_ctr = anchor[1] + 0.5 * (h - 1);

	for (std::size_t s = 0; s < scales.size(); ++s) {
	    float ws = w * scales[s];
	    float hs = h * scales[s];
	    scale_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
			y_ctr - 0.5 * (hs - 1),
			x_ctr + 0.5 * (ws - 1),
			y_ctr + 0.5 * (hs - 1)));
	}
    }
}


void AnchorGenerator::bbox_pred(const CRect2f& anchor, const CRect2f& delta, CRect2f& box) const{
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    float dx = delta[0];
    float dy = delta[1];
    float dw = delta[2];
    float dh = delta[3];

    float pred_ctr_x = dx * w + x_ctr;
    float pred_ctr_y = dy * h + y_ctr;
    float pred_w = std::exp(dw) * w;
    float pred_h = std::exp(dh) * h;

    box = CRect2f((pred_ctr_x - 0.5 * (pred_w - 1.0))*ratiow,
	    (pred_ctr_y - 0.5 * (pred_h - 1.0))*ratioh,
	    (pred_ctr_x + 0.5 * (pred_w - 1.0))*ratiow,
	    (pred_ctr_y + 0.5 * (pred_h - 1.0))*ratioh);
}


void AnchorGenerator::landmark_pred(const CRect2f& anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts) const{
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = anchor[0] + 0.5 * (w - 1);
    float y_ctr = anchor[1] + 0.5 * (h - 1);

    pts.resize(delta.size());
    for (std::size_t i = 0; i < delta.size(); ++i) {
        pts[i].x = (delta[i].x*w + x_ctr)*ratiow;
        pts[i].y = (delta[i].y*h + y_ctr)*ratioh;
    }
}



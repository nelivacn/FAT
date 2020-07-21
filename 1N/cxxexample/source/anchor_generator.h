#ifndef FAT_ANCHOR_GENERTOR_H
#define FAT_ANCHOR_GENERTOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "common.hpp"

namespace FAT {


//class Anchor {
//    public:
//	Anchor() {
//	}
//
//	~Anchor() {
//	}
//
//	bool operator<(const Anchor &t) const {
//	    return score < t.score;
//	}
//
//	bool operator>(const Anchor &t) const {
//	    return score > t.score;
//	}
//
//	float& operator[](int i) {
//	    assert(0 <= i && i <= 4);
//
//	    if (i == 0)
//		return finalbox.x;
//	    if (i == 1)
//		return finalbox.y;
//	    if (i == 2)
//		return finalbox.width;
//	    if (i == 3)
//		return finalbox.height;
//	}
//
//	float operator[](int i) const {
//	    assert(0 <= i && i <= 4);
//
//	    if (i == 0)
//		return finalbox.x;
//	    if (i == 1)
//		return finalbox.y;
//	    if (i == 2)
//		return finalbox.width;
//	    if (i == 3)
//		return finalbox.height;
//	}
//
//	cv::Rect2f anchor; // x1,y1,x2,y2
//	float reg[4]; // offset reg
//	cv::Point center; // anchor feat center
//	float score; // cls score
//	std::vector<cv::Point2f> pts; // pred pts
//
//	cv::Rect2f finalbox; // final box res
//
//	void print() {
//	    printf("finalbox %f %f %f %f, score %f\n", finalbox.x, finalbox.y, finalbox.width, finalbox.height, score);
//	    printf("landmarks ");
//	    for (int i = 0; i < pts.size(); ++i) {
//		printf("%f %f, ", pts[i].x, pts[i].y);
//	    }
//	    printf("\n");
//	}
//};

class AnchorCfg {
public:
	  std::vector<float> SCALES;	
	  std::vector<float> RATIOS;
	  int BASE_SIZE;

      AnchorCfg() {}
      ~AnchorCfg() {}
	  AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size) {
			  SCALES = s;
			  RATIOS = r;
			  BASE_SIZE = size;
	  }
};

class AnchorGenerator {
    public:
	AnchorGenerator();
	~AnchorGenerator();

	// init different anchors
	void Init(int stride, const AnchorCfg& cfg);

	void Predict(const cv::Mat& cls, const cv::Mat& reg, const cv::Mat& pts, float score_thresh, std::vector<Face>& result) const;

    private:
	void _ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors);

	void _scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors);

	void bbox_pred(const CRect2f& anchor, const CRect2f& delta, CRect2f& box) const;

	void landmark_pred(const CRect2f& anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts) const;

	//std::vector<std::vector<Anchor>> anchor_planes; // corrspont to channels

	std::vector<int> anchor_size;
	std::vector<float> anchor_ratio;
	float anchor_step_; // scale step
	int anchor_stride_; // anchor tile stride
	int feature_w; // feature map width
	int feature_h; // feature map height

	std::vector<CRect2f> preset_anchors_;
	int anchor_num_; // anchor type num
    float score_thresh_;

	float ratiow=1.0;
	float ratioh=1.0;

};

}

#endif // ANCHOR_GENERTOR

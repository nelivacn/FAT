/*!
 * Copyright (c) 2020 by NELIVA
 * \file common.hpp
 * \brief
 * \author Jia Guo
*/
#ifndef FAT_COMMON_HPP
#define FAT_COMMON_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace FAT {
}
class CRect2f {
    public:
	CRect2f() { 
    }
	CRect2f(float x1, float y1, float x2, float y2) {
	    val[0] = x1;
	    val[1] = y1;
	    val[2] = x2;
	    val[3] = y2;
	}

	float& operator[](int i) {
	    return val[i];
	}

	float operator[](int i) const {
	    return val[i];
	}

	float val[4] = {0.0, 0.0, 0.0, 0.0};

	//void print() {
	//    printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
	//}
};

typedef CRect2f BBox;


class Face {
    public:
        Face() : score(-1.0) {

        }
        CRect2f bbox;
        float score; // cls score
        std::vector<cv::Point2f> pts; // pred pts
        //std::vector<float> feature;

};

#endif 


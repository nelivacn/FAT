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

namespace FAT {

class FATImpl {
    public:
        FATImpl();
        ~FATImpl();

        static int FeatureLength() { return Feature::FeatureLength(); }

        void Load(const std::string& rdir);

        void GetFeature(char* img, int width, int height, int channel, int im_type, float* feat);

        float GetSim(float* feat1, float* feat2);

    private:
        std::shared_ptr<Detector> detector_;
        std::shared_ptr<Feature> feature_;

};

}

#endif

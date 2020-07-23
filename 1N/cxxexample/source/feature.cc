/*!
 * Copyright (c) 2020 by NELIVA
 * \file common.hpp
 * \brief
 * \author Jia Guo
*/
#include "feature.h"

using namespace FAT;

cv::Mat meanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i ++)
    {
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++)
        {
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}

cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}


cv::Mat varAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return meanAxis0(temp_);

}



int MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}

cv::Mat similarTransform(cv::Mat src,cv::Mat dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S,U, V);

    // the SVD function in opencv differ from scipy .


    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
//            s = d[dim - 1]
//            d[dim - 1] = -1
//            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
//            d[dim - 1] = s
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U* twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else{
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U* twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d,S,res);
    float scale =  1.0/val*cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat  temp2 = src_mean.t(); //src_mean.T
    cv::Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale*temp3;
    T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    T = T.rowRange(0,2);
    return T;
}

Feature::Feature(const std::string& onnx_file) {
    std::cout<<"Loading: "<<onnx_file<<std::endl;
    net_ = cv::dnn::readNetFromONNX(onnx_file);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    float v[5][2] = {
      {38.2946f, 51.6963f},
      {73.5318f, 51.5014f},
      {56.0252f, 71.7366f},
      {41.5493f, 92.3655f},
      {70.7299f, 92.2041f}}; //112x112
    dst_ = cv::Mat(5,2,CV_32FC1);
    memcpy(dst_.data, v, 2 * 5 * sizeof(float));
}

void Feature::Get(const cv::Mat& img, const Face& face, float* feat) {
    //cv::imwrite("./raw.png", img);
    cv::Mat input;
    cv::Size input_size(112, 112);
    if(face.pts.empty()) {
        cv::resize(img, input, input_size);
    }
    else {
        //std::cout<<img.cols<<","<<img.rows<<std::endl;
        //for(int i=0;i<4;i++) {
        //    std::cout<<face.bbox[i]<<std::endl;
        //}
        float v[5][2];
        for(int i=0;i<5;i++) {
            v[i][0] = face.pts[i].x;
            v[i][1] = face.pts[i].y;
            //std::cout<<"pts"<<i<<":"<<face.pts[i].x<<","<<face.pts[i].y<<std::endl;
        }

        cv::Mat src(5,2,CV_32FC1, v);
        //memcpy(src.data, v, 2 * 5 * sizeof(float));
        //std::cout<<src<<std::endl;
        //std::cout<<dst_<<std::endl;
        cv::Mat trans_mat = similarTransform(src,dst_);
        cv::warpAffine(img, input, trans_mat, input_size); 
    }
    //cv::imwrite("./aligned.png", input);
    auto blob = cv::dnn::blobFromImage(input, 1.0, input_size, cv::Scalar(0.0, 0.0, 0.0), true, false);
    cv::Mat feature;
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    //std::cout<<output.total()<<std::endl;
    memcpy(feat, (float*)output.data, sizeof(float)*output.total());
    float norm = 0.00001;
    for(std::size_t i=0;i<output.total();i++) {
        norm += feat[i]*feat[i];
    }
    for(std::size_t i=0;i<output.total();i++) {
        feat[i] /= std::sqrt(norm);
    }
}


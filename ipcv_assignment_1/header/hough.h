//
// Created by George Lancaster on 14/11/2018.
//

#ifndef OPENCV_HOUGH_H
#define OPENCV_HOUGH_H

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat getYKernel();

Mat getXKernel();

int ***allocate3DArray(int x, int y, int z);

vector<Vec2f> hough_line(const Mat &src, int threshold, int delta);

vector<Vec3f> hough_circle(const Mat &src, int threshold, int minRadius, int maxRadius);

Mat gradMagnitude(Mat input);

Mat gradDirection(Mat input);

void thresholdMag(Mat &image, int threshold);


#endif //OPENCV_HOUGH_H
